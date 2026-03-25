/* =====================================================
   PoseDetector — Multi-Person MoveNet via TensorFlow.js
   
   UPGRADE: SinglePose → MultiPose Lightning
   Detects up to 6 people simultaneously.
   Falls back to ml5.js PoseNet (single-person) if needed.
   ===================================================== */

export class PoseDetector {
    constructor() {
        this.detector = null;
        this.model = null;
        this.isReady = false;
        this.frameCount = 0;
        this.fps = 0;
        this._lastFPSTime = 0;
        this._fpsFrameCount = 0;
        this.isMultiPose = false;
        this.config = {
            model: 'MoveNet',
            variant: 'MultiPose',
            minScore: 0.25,
            maxPoses: 6,
        };
    }

    async init(video, onProgress = null) {
        this.video = video;
        if (onProgress) onProgress(10);

        if (typeof poseDetection !== 'undefined') {
            // Try MultiPose first, then SinglePose, then PoseNet
            try { return await this._initMoveNetMulti(onProgress); }
            catch (err) {
                console.warn('MoveNet MultiPose failed, trying SinglePose:', err);
                try { return await this._initMoveNetSingle(onProgress); }
                catch (err2) { console.warn('MoveNet SinglePose also failed:', err2); }
            }
        }
        if (typeof ml5 !== 'undefined') {
            return await this._initPoseNet(onProgress);
        }
        throw new Error('No pose detection library available.');
    }

    async _initMoveNetMulti(onProgress) {
        if (onProgress) onProgress(20);
        await tf.ready();
        if (onProgress) onProgress(40);

        const modelType = poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING;
        if (onProgress) onProgress(50);

        this.detector = await poseDetection.createDetector(
            poseDetection.SupportedModels.MoveNet,
            {
                modelType,
                enableSmoothing: true,
                enableTracking: true,
                trackerType: poseDetection.TrackerType.BoundingBox,
                minPoseScore: this.config.minScore,
            }
        );
        this.model = 'MoveNet';
        this.isMultiPose = true;
        this.isReady = true;
        if (onProgress) onProgress(100);
        console.log('✅ MoveNet MultiPose Lightning loaded (up to 6 people)');
        return 'MoveNet MultiPose';
    }

    async _initMoveNetSingle(onProgress) {
        if (onProgress) onProgress(20);
        await tf.ready();
        if (onProgress) onProgress(40);

        const modelType = poseDetection.movenet.modelType.SINGLEPOSE_THUNDER;
        if (onProgress) onProgress(50);

        this.detector = await poseDetection.createDetector(
            poseDetection.SupportedModels.MoveNet,
            { modelType, enableSmoothing: true, minPoseScore: this.config.minScore }
        );
        this.model = 'MoveNet';
        this.isMultiPose = false;
        this.isReady = true;
        if (onProgress) onProgress(100);
        console.log('✅ MoveNet SinglePose Thunder loaded');
        return 'MoveNet Single';
    }

    async _initPoseNet(onProgress) {
        if (onProgress) onProgress(30);
        return new Promise((resolve) => {
            const opts = {
                architecture: 'MobileNetV1', imageScaleFactor: 0.3,
                outputStride: 16, flipHorizontal: false,
                minConfidence: this.config.minScore, detectionType: 'multiple',
                maxPoseDetections: this.config.maxPoses,
            };
            this._poseNetModel = ml5.poseNet(this.video, opts, () => {
                this.model = 'PoseNet';
                this.isMultiPose = true;
                this.isReady = true;
                if (onProgress) onProgress(100);
                console.log('✅ PoseNet loaded (fallback, multi-person)');
                resolve('PoseNet Multi');
            });
            this._poseNetResults = null;
            this._poseNetModel.on('pose', (r) => { this._poseNetResults = r; });
        });
    }

    /**
     * Detect all poses in the current frame.
     * @returns {Object[]|null} Array of {keypoints, score} objects, one per person
     */
    async detectAll() {
        if (!this.isReady) return null;
        this._updateFPS();

        if (this.model === 'MoveNet') return await this._detectMoveNetAll();
        return this._detectPoseNetAll();
    }

    /** Legacy single-pose API (returns first person only) */
    async detect() {
        const all = await this.detectAll();
        return all && all.length > 0 ? all[0] : null;
    }

    async _detectMoveNetAll() {
        try {
            const poses = await this.detector.estimatePoses(this.video, {
                flipHorizontal: false,
                maxPoses: this.config.maxPoses,
            });
            if (!poses || poses.length === 0) return null;
            this.frameCount++;

            return poses
                .filter(pose => (pose.score || this._avgScore(pose.keypoints)) > this.config.minScore)
                .map(pose => ({
                    keypoints: pose.keypoints.map(kp => ({
                        x: kp.x, y: kp.y, score: kp.score, name: kp.name,
                    })),
                    score: pose.score || this._avgScore(pose.keypoints),
                    id: pose.id,       // MoveNet MultiPose provides built-in tracking ID
                }));
        } catch (err) {
            console.error('Detection error:', err);
            return null;
        }
    }

    _detectPoseNetAll() {
        if (!this._poseNetResults || this._poseNetResults.length === 0) return null;
        this.frameCount++;

        return this._poseNetResults
            .filter(r => r.pose && r.pose.score > this.config.minScore)
            .map(r => ({
                keypoints: r.pose.keypoints.map(kp => ({
                    x: kp.position.x, y: kp.position.y, score: kp.score, name: kp.part,
                })),
                score: r.pose.score || 0,
            }));
    }

    _avgScore(keypoints) {
        const v = keypoints.filter(kp => (kp.score || 0) > 0.1);
        if (v.length === 0) return 0;
        return v.reduce((s, kp) => s + (kp.score || 0), 0) / v.length;
    }

    _updateFPS() {
        this._fpsFrameCount++;
        const now = performance.now();
        if (now - this._lastFPSTime >= 1000) {
            this.fps = Math.round((this._fpsFrameCount * 1000) / (now - this._lastFPSTime));
            this._fpsFrameCount = 0;
            this._lastFPSTime = now;
        }
    }

    dispose() {
        if (this.detector && this.detector.dispose) this.detector.dispose();
        this.isReady = false;
    }

    getInfo() {
        return {
            model: this.model || 'N/A',
            variant: this.config.variant,
            multiPose: this.isMultiPose,
            maxPoses: this.config.maxPoses,
            fps: this.fps,
            frames: this.frameCount,
            ready: this.isReady,
        };
    }
}
