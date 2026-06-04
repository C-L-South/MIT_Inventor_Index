const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

//for alerting
let missingBodySince = null;
let alertSent = false;
let startSignal=false;
let detector = null;
let animationId = null;
let streamRef = null;
let running = false;

// ── rep counter persistent state ──────────────────────
let rep_initialized = false;
let count_p = 0;
let state_p = 0;
let angleDiffFilt_p = 0;
let fastSlowWarning_p = 0;
let highAngle = NaN;
let lowAngle = NaN;
let highTime = NaN;
let lowTime = NaN;
let rep_prev_angle = null;
function getExeRepCfg(caseId) {
  switch (caseId) {
    case 1: // squat
      return {
        alpha:         0.20,
        highThresh:    2.5,
        lowThresh:    -2.5,
        minAmp:        5.0,
        minPeriod:     0.4,
        maxPeriod:     5.0,
        startTime:     10.0,
        minPeriodWrn:  1.0,
        maxPeriodWrn:  4.0,
      };
    default:
      return null;
  }
}
const cfg = getExeRepCfg(1);


function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawKeypoints(keypoints, scale, offsetX, offsetY, warningColor) {
  const pointColor = warningColor || "#00ff8a";

  for (const kp of keypoints) {
    if (kp.score > 0.3) {
      const x = kp.x * scale + offsetX;
      const y = kp.y * scale + offsetY;

      ctx.beginPath();
      ctx.arc(x, y, 13, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = pointColor;
      ctx.fill();
    }
  }
}

function drawSkeleton(keypoints, scale, offsetX, offsetY, warningColor) {
  const adjacentPairs = poseDetection.util.getAdjacentPairs(
    poseDetection.SupportedModels.MoveNet
  );

  ctx.strokeStyle = warningColor || "white";
  ctx.lineWidth = 5;

  for (const [i, j] of adjacentPairs) {
    const kp1 = keypoints[i];
    const kp2 = keypoints[j];

    if (kp1 && kp2 && kp1.score > 0.3 && kp2.score > 0.3) {
      ctx.beginPath();
      ctx.moveTo(kp1.x * scale + offsetX, kp1.y * scale + offsetY);
      ctx.lineTo(kp2.x * scale + offsetX, kp2.y * scale + offsetY);
      ctx.stroke();
    }
  }
}

async function setupDetector() {
    if (detector) return detector;
    await tf.ready();
    detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER }
    );
    return detector;
}

async function detectPose() {
    if (!running) return;
    try {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const poses = await detector.estimatePoses(video);

    if (poses.length > 0) {
        const keypoints = poses[0].keypoints || [];

        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;
        const scale = Math.max(scaleX, scaleY);
        const offsetX = (canvas.width  - video.videoWidth  * scale) / 2;
        const offsetY = (canvas.height - video.videoHeight * scale) / 2;

        const allPointsVisible = keypoints.every(kp => kp.score > 0.3);
        const warningColor = allPointsVisible ? null : "orange";

        if (!allPointsVisible) {

          if (missingBodySince === null) {
            missingBodySince = Date.now();
          }

          if (
            !alertSent &&
            Date.now() - missingBodySince >= 2000 &&
            window.AppInventor
          ) {
            alertSent = true;
            window.AppInventor.setWebViewString(
              "Please move your body so it is visible in the camera."
            );
          }

        } else {

          // reset when body is visible again
          missingBodySince = null;
          alertSent = false;

        }
        if (window.AppInventor && startSignal==false) {
          window.AppInventor.setWebViewString("Movenet Starting");
          startSignal=true;
        }
        drawSkeleton(keypoints, scale, offsetX, offsetY, warningColor);
        drawKeypoints(keypoints, scale, offsetX, offsetY, warningColor);


        const row = formatMoveNetPointsForMatlab(poses[0]);
        compute(row);
    }

    animationId = requestAnimationFrame(detectPose);
    } catch (error) {
    console.error(error);
    stopCamera();
    }
}

async function startCamera() {
    if (running) return;
    try {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            backgroundBlur: true
        },
        audio: false
    });
    streamRef = stream;
    video.srcObject = stream;
    await new Promise((resolve) => { video.onloadedmetadata = resolve; });
    await video.play();
    resizeCanvas();

    await setupDetector();
 
    running = true;

    //starts detection
    detectPose();
    } catch (error) {
    console.error(error);
    }
}

function stopCamera() {
    running = false;

    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }

    if (streamRef) {
        streamRef.getTracks().forEach(t => t.stop());
        streamRef = null;
    }

    video.srcObject = null;

    video.style.display = "none";
    canvas.style.display = "none";

    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function getJointAngle(threePoints) {
  // threePoints should be:
  // [
  //   [x1, y1],
  //   [x2, y2],  // middle/reference point
  //   [x3, y3]
  // ]

  const eps = 1e-6;

  let vector1 = [
    threePoints[0][0] - threePoints[1][0],
    threePoints[0][1] - threePoints[1][1],
  ];

  let vector2 = [
    threePoints[2][0] - threePoints[1][0],
    threePoints[2][1] - threePoints[1][1],
  ];

  const norm1 = Math.sqrt(vector1[0] ** 2 + vector1[1] ** 2);
  const norm2 = Math.sqrt(vector2[0] ** 2 + vector2[1] ** 2);

  if (norm1 < eps || norm2 < eps) {
    return 0;
  }

  vector1 = [vector1[0] / norm1, vector1[1] / norm1];
  vector2 = [vector2[0] / norm2, vector2[1] / norm2];

  const dotProduct = vector1[0] * vector2[0] + vector1[1] * vector2[1];
  const clamped = Math.max(-1, Math.min(1, dotProduct)); // clamp to [-1, 1]
  return Math.acos(clamped); // ✅ never NaN
}
function formatMoveNetPointsForMatlab(pose) {
  const k = pose.keypoints;

  const timestamp = Date.now();

  const pointMap = {
    1: "left_ankle",
    2: "right_ankle",
    3: "left_wrist",
    4: "right_wrist",
    5: "left_hip",
    6: "right_hip",
    7: "left_knee",
    8: "right_knee",
    9: "left_shoulder",
    10: "right_shoulder",
    11: "left_elbow",
    12: "right_elbow",
    13: "nose"
  };

  const row = [timestamp];

  for (let i = 1; i <= 13; i++) {
    const name = pointMap[i];
    const pt = k.find(p => p.name === name);

    if (pt && pt.score > 0.3) {
      row.push(Math.round(pt.x));
      row.push(Math.round(pt.y));
    } else {
      row.push(-1);
      row.push(-1);
    }
  }
  document.getElementById("output").textContent =
    `[${row.join(", ")}]`;

  return row;
}
function getPoint(row, idx) {
  return [
    row[1 + (idx - 1) * 2],
    row[2 + (idx - 1) * 2]
  ];
}
function count_exercise_rep(curTime, angle, reset, cfg) {

    if (reset || !rep_initialized) {
        rep_initialized   = true;
        count_p           = 0;
        state_p           = 0;
        angleDiffFilt_p   = 0;
        fastSlowWarning_p = 0;
        highAngle         = NaN;
        lowAngle          = NaN;
        highTime          = NaN;
        lowTime           = NaN;
        rep_prev_angle    = angle;
        return [count_p, state_p, angleDiffFilt_p, fastSlowWarning_p];
    }

    // filter angle diff
    const angleDiff = angle - rep_prev_angle;
    angleDiffFilt_p = cfg.alpha * angleDiff + (1 - cfg.alpha) * angleDiffFilt_p;

    // ignore startup transient
    if (curTime < cfg.startTime) {
        rep_prev_angle = angle;
        return [count_p, state_p, angleDiffFilt_p, fastSlowWarning_p];
    }

    // state machine
    if (state_p === 0) {
        if (angleDiffFilt_p > cfg.highThresh) {
            highAngle = angleDiffFilt_p;
            highTime  = curTime;
            state_p   = 1;
        }
    } else if (state_p === 1) {
        if (angleDiffFilt_p < cfg.lowThresh) {
            lowAngle = angleDiffFilt_p;
            lowTime  = curTime;
            state_p  = 2;
        }
    } else if (state_p === 2) {
        if (angleDiffFilt_p > cfg.highThresh) {
            const repPeriod = curTime - highTime;
            const repAmp    = Math.abs(highAngle - lowAngle);

            if (repAmp >= cfg.minAmp &&
                repPeriod >= cfg.minPeriod &&
                repPeriod <= cfg.maxPeriod) {

                count_p++;

                if (repPeriod >= cfg.maxPeriodWrn) {
                    fastSlowWarning_p = 2;
                } else if (repPeriod <= cfg.minPeriodWrn) {
                    fastSlowWarning_p = 1;
                } else {
                    fastSlowWarning_p = 0;
                }
            }

            highAngle = angleDiffFilt_p;
            highTime  = curTime;
            state_p   = 1;
        }
    }

    rep_prev_angle = angle;
    return [count_p, state_p, angleDiffFilt_p, fastSlowWarning_p];
}

function hasInvalidPoint(points) {
  return points.some(pt => pt[0] < 0 || pt[1] < 0);
}
const DTOR = Math.PI / 180;
const RTOD = 180 / Math.PI;

const TEMPLATE = [145, 30, 145, 30].map(v => v * DTOR);
const TEMPLATE_NORM = Math.hypot(...TEMPLATE);

let prvTime = null;
let prvAng1 = Math.PI;
let prvAng2 = Math.PI;

let ang1_x = null;
let ang1_x2 = null;
let ang1_sigma = 0;

let ang2_x = null;
let ang2_x2 = null;
let ang2_sigma = 0;

const tau= 3;
function compute(row){
    const pt6 = getPoint(row, 6);
    const pt8 = getPoint(row, 8);
    const pt2 = getPoint(row, 2);

    let allThreePts = [pt6, pt8, pt2];

    let ang1;
    if (!hasInvalidPoint(allThreePts)) {
    ang1 = getJointAngle(allThreePts);
    prvAng1 = ang1;
    } else {
    ang1 = prvAng1;
    }

    const pt5 = getPoint(row, 5);
    const pt7 = getPoint(row, 7);
    const pt1 = getPoint(row, 1);

    allThreePts = [pt5, pt7, pt1];

    let ang2;
    if (!hasInvalidPoint(allThreePts)) {
    ang2 = getJointAngle(allThreePts);
    prvAng2 = ang2;
    } else {
    ang2 = prvAng2;
    }

    const timeStamp = row[0] / 1000;

    if (prvTime !== null) {
    const dt = timeStamp - prvTime;
    const alpha = tau / (dt + tau);

    ang1_x = alpha * ang1_x + (1 - alpha) * ang1;
    ang1_x2 = alpha * ang1_x2 + (1 - alpha) * ang1 ** 2;
    ang1_sigma = Math.sqrt(Math.max(0, ang1_x2 - ang1_x ** 2));

    ang2_x = alpha * ang2_x + (1 - alpha) * ang2;
    ang2_x2 = alpha * ang2_x2 + (1 - alpha) * ang2 ** 2;
    ang2_sigma = Math.sqrt(Math.max(0, ang2_x2 - ang2_x ** 2));
    } else {
    ang1_x = ang1;
    ang1_x2 = ang1 ** 2;
    ang1_sigma = 0;

    ang2_x = ang2;
    ang2_x2 = ang2 ** 2;
    ang2_sigma = 0;
    }

    prvTime = timeStamp;

    const featureVect = [ang1_x, ang1_sigma, ang2_x, ang2_sigma];

    const featureNorm = Math.hypot(...featureVect);

    const normalizedFeature = featureVect.map(v => v / TEMPLATE_NORM);

    const normalizedTemplate = TEMPLATE.map(v => v / TEMPLATE_NORM);

    const similarity = Math.hypot(
    ...normalizedFeature.map((v, i) => v - normalizedTemplate[i])
    );
    const bestError = 0.01;
    const worstError = 0.3113;

    const accuracyScore = Math.max(
      0,
      Math.min(
        100,
        ((worstError - similarity) / (worstError - bestError)) * 100
      )
    );
    const ang1Deg = ang1 * RTOD;
    
    const [count_L, state_L, angleDiffFilt_L, fastSlowWrn_L] = count_exercise_rep(
      timeStamp,
      ang1Deg,
      0,
      cfg
    );
    if (window.AppInventor) {
      window.AppInventor.setWebViewString(count_L);
    }
    document.getElementById("output").textContent =
    `row:
    [${row.join(", ")}]

    ang1: ${(ang1 * RTOD).toFixed(1)} deg
    ang2: ${(ang2 * RTOD).toFixed(1)} deg

    featureVect:
    [
    ${featureVect.map(v => (v * RTOD).toFixed(2)).join(", ")}
    ]

    similarity:
    ${similarity.toFixed(4)}`;

    const bar = document.getElementById("similarityBar");

    bar.style.width = `${accuracyScore}%`;

    if (accuracyScore >= 66.67) {
      bar.style.background = "lime";
    } else if (accuracyScore >= 33.33) {
      bar.style.background = "yellow";
    } else {
      bar.style.background = "red";
    }

}



// App Inventor control
function receiveFromApp(command) {
    if (command === 'start') startCamera();
    else if (command === 'stop') stopCamera();
}
startCamera();
