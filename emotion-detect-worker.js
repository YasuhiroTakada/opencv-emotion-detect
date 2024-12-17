importScripts("./lib/opencv.js");
importScripts("./worker-util.js");

const url = "/models/emotion-ferplus-12-int8.onnx";

self.onmessage = (message) => {
    runWithInitialize(() => {
        const [rows, cols, bytes, index] = [...message.data];
        console.time(`emotion-worker-${index}`);
        const srcMat = new cv.matFromArray(rows, cols, cv.CV_8UC4, bytes);
        const result = self.run(srcMat);
        console.timeEnd(`emotion-worker-${index}`);
        self.postMessage([index, result]);
        srcMat.delete();
    });
}

let net;
runWithInitialize = (callback) => {
    if (typeof cv !== 'undefined') {
        if (cv.getBuildInformation) {
            if (self.net) {
                callback();
            } else {
                loadModel(cv, url)
                    .then((net) => self.net = net)
                    .then(() => callback());
            }
        } else {
            cv['onRuntimeInitialized'] = () => {
                runWithInitialize(callback);
            };
        }
    } else {
        console.error('Failed to load OpenCV.js');
    }
}

function run(faceMat) {
    if (!self.net) {
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

    self.net.setInput(inputBlob);
    const output = self.net.forward();

    const total = output.data32F.reduce((acc, val) => acc + Math.max(val, 0), 0);
    const emotions = {
        neutral: Math.max(output.data32F[0], 0) / total,
        happiness: Math.max(output.data32F[1], 0) / total,
        surprise: Math.max(output.data32F[2], 0) / total,
        sadness: Math.max(output.data32F[3], 0) / total,
        anger: Math.max(output.data32F[4], 0) / total,
        disgust: Math.max(output.data32F[5], 0) / total,
        fear: Math.max(output.data32F[6], 0) / total,
        contempt: Math.max(output.data32F[7], 0) / total,
    };

    inputBlob.delete();
    output.delete();

    return {emotions};
}
