import {loadModel} from "./util.js";

let emotionNet;

export async function setup() {
    emotionNet = await loadModel("https://media.githubusercontent.com/media/onnx/models/refs/heads/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-12-int8.onnx?download=true")
}

export function detectEmotion(detections, imageWidth, imageHeight, src) {
    return detections.map((det, index) => {
        const faceMat = src.roi(new cv.Rect(Math.round(det.x_min), Math.round(det.y_min), Math.round(det.x_max - det.x_min), Math.round(det.y_max - det.y_min)));
        const emotions = detectEmotions(faceMat);
        faceMat.delete();

        return {
            faceId: `face-${index + 1}`,
            faceRectangle: {
                top: Math.round(det.y_min),
                left: Math.round(det.x_min),
                width: Math.round(det.x_max - det.x_min),
                height: Math.round(det.y_max - det.y_min),
            },
            // confidence: det.score.toFixed(2),
            emotions,
        };
    });
}

function detectEmotions(faceMat) {
    if (!emotionNet) {
        console.error('Emotion model not loaded.');
        return {};
    }

    const inputBlob = cv.blobFromImage(
        faceMat,
        1.0,
        new cv.Size(64, 64), // Model input size for emotions
        new cv.Scalar(0, 0, 0),
        true,
        false
    );

    emotionNet.setInput(inputBlob);
    const output = emotionNet.forward();
    inputBlob.delete();

    const emotions = {
        neutral: output.data32F[0],
        happiness: output.data32F[1],
        surprise: output.data32F[2],
        sadness: output.data32F[3],
        anger: output.data32F[4],
        disgust: output.data32F[5],
        fear: output.data32F[6],
        contempt: output.data32F[7],
    };
    output.delete();
    return emotions;
}
