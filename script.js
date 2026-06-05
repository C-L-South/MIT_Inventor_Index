if (poses.length > 0) {
    const keypoints = poses[0].keypoints || [];

    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    const scale = Math.max(scaleX, scaleY);
    const offsetX = (canvas.width  - video.videoWidth  * scale) / 2;
    const offsetY = (canvas.height - video.videoHeight * scale) / 2;

    const now = Date.now();

    // Check whether all points are visible
    const allPointsVisible = keypoints.every(kp => kp.score > 0.3);

    // If all points visible, update last good pose time
    if (allPointsVisible) {
        lastGoodPoseTime = now;
    }

    // Allow brief tracking losses (500 ms)
    const poseRecentlyGood = (now - lastGoodPoseTime) < 500;

    // Only show warning color if body has been missing longer than grace period
    const warningColor = poseRecentlyGood ? null : "orange";

    if (!poseRecentlyGood) {

        if (missingBodySince === null) {
            missingBodySince = now;
        }

        if (
            !alertSent &&
            now - missingBodySince >= 2000 &&
            window.AppInventor
        ) {
            alertSent = true;
            window.AppInventor.setWebViewString(
                "Please move your body so it is visible in the camera."
            );
        }

    } else {

        missingBodySince = null;
        alertSent = false;
    }
