// Wait for OpenCV.js to load
import {loadModel} from "./util.js";

let detectNet;

export async function setup() {
    detectNet = await loadModel("https://media.githubusercontent.com/media/onnx/models/refs/heads/main/validated/vision/body_analysis/ultraface/models/version-RFB-320-int8.onnx?download=true")
}

export function detectFaces(srcMat) {
    if (!detectNet) {
        console.error(
            'Model is not loaded yet. Please wait for OpenCV.js to initialize.'
        );
        return;
    }
    // Convert the image to a Mat
    let matC3 = new cv.Mat(srcMat.matSize[0], srcMat.matSize[1], cv.CV_8UC3);
    cv.cvtColor(srcMat, matC3, cv.COLOR_RGBA2BGR);
    // Prepare the input blob
    const inputBlob = cv.blobFromImage(
        matC3,
        1 / 128, // Scale factor
        new cv.Size(320, 240), // Model input size
        new cv.Scalar(127, 127, 127),
        true, // Swap RB
        false // Crop
    );
    // Set input and run inference
    detectNet.setInput(inputBlob);
    const scores = detectNet.forward("scores");
    const boxes = detectNet.forward("boxes");

    const scoresData = scores.data32F;
    const boxesData = boxes.data32F;
    const detections = [];

    const numCandidates = scoresData.length / 2; // 4420 anchors in this model

    for (let i = 0; i < numCandidates; i++) {
        const faceScore = scoresData[i * 2 + 1]; // Index 1 for face class confidence
        if (faceScore > 0.7) { // Threshold for face detection
            const x_min = boxesData[i * 4 + 0] * srcMat.cols;
            const y_min = boxesData[i * 4 + 1] * srcMat.rows;
            const x_max = boxesData[i * 4 + 2] * srcMat.cols;
            const y_max = boxesData[i * 4 + 3] * srcMat.rows;
            detections.push({ x_min, y_min, x_max, y_max });
        }
    }

    const nmsDetections = applyNMS(detections, 0.4); // IoU threshold of 0.4

    // Draw detections on the canvas
    // nmsDetections.forEach((det) => {
    //     ctx.strokeStyle = 'red';
    //     ctx.lineWidth = 2;
    //     ctx.strokeRect(det.x_min, det.y_min, det.x_max - det.x_min, det.y_max - det.y_min);
    // });

    // Cleanup
    inputBlob.delete();
    scores.delete();
    boxes.delete();

    return nmsDetections;
}

function extractFaceMats(img, detections) {
    return detections.map((det, index) =>
        src.roi(new cv.Rect(Math.round(det.x_min), Math.round(det.y_min), Math.round(det.x_max - det.x_min), Math.round(det.y_max - det.y_min))));
}

function applyNMS(detections, iouThreshold) {
    // Sort detections by score in descending order
    detections.sort((a, b) => b.score - a.score);

    const nmsDetections = [];

    while (detections.length > 0) {
        const best = detections.shift();
        nmsDetections.push(best);

        detections = detections.filter((det) => {
            const iou = calculateIoU(best, det);
            return iou < iouThreshold;
        });
    }

    return nmsDetections;
}

function calculateIoU(boxA, boxB) {
    const x1 = Math.max(boxA.x_min, boxB.x_min);
    const y1 = Math.max(boxA.y_min, boxB.y_min);
    const x2 = Math.min(boxA.x_max, boxB.x_max);
    const y2 = Math.min(boxA.y_max, boxB.y_max);

    const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const boxAArea = (boxA.x_max - boxA.x_min) * (boxA.y_max - boxA.y_min);
    const boxBArea = (boxB.x_max - boxB.x_min) * (boxB.y_max - boxB.y_min);

    const unionArea = boxAArea + boxBArea - intersectionArea;

    return intersectionArea / unionArea;
}