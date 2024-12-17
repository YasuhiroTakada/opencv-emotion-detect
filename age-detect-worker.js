importScripts("./lib/opencv.js");
importScripts("./worker-util.js");

const url = "/models/age_googlenet.onnx";

const ageList = [[0, 2], [4, 6], [8, 12], [15, 20], [25, 32], [38, 43], [48, 53], [60, 100]];

let net;

self.onmessage = (message) => {
    runWithInitialize(() => {
        const [rows, cols, bytes, index] = [...message.data];
        console.time(`age-worker-${index}`);
        const srcMat = new cv.matFromArray(rows, cols, cv.CV_8UC4, bytes);
        const result = self.run(srcMat);
        console.timeEnd(`age-worker-${index}`);
        self.postMessage([index, result]);
        srcMat.delete();
    });
}

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

// see
// https://github.com/onnx/models/blob/main/validated/vision/body_analysis/age_gender/levi_googlenet.py
function run(srcMat) {
    if (!self.net) {
        console.error('Age model not loaded.');
        return {};
    }

    const matC3 = new cv.Mat(srcMat.matSize[0], srcMat.matSize[1], cv.CV_8UC3);
    cv.cvtColor(srcMat, matC3, cv.COLOR_RGBA2BGR);

    const inputBlob = cv.blobFromImage(
        matC3,
        1.0,
        new cv.Size(224, 224), // Model input size for emotions
        new cv.Scalar(104, 117, 123),
        false,
        false
    );

    self.net.setInput(inputBlob);
    const output = self.net.forward();

    const age = { age: ageList[argMax(output.data32F)], source: output.data32F };

    output.delete();
    inputBlob.delete();
    matC3.delete();

    return age;
}
