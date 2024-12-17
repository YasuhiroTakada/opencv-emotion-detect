importScripts("../lib/opencv.js");
importScripts("./worker-util.js");

const url = "/models/gender_googlenet.onnx";

const genderList = ['Male', 'Female'];

self.onmessage = (message) => {
    runWithInitialize(() => {
        const [rows, cols, bytes, index] = [...message.data];
        console.time(`gender-worker-${index}`);
        const srcMat = new cv.matFromArray(rows, cols, cv.CV_8UC4, bytes);
        const result = self.run(srcMat);
        self.postMessage([index, result]);
        srcMat.delete();
        console.timeEnd(`gender-worker-${index}`);
    });
}

let net;
// download model and cache.
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

// reference 
// https://github.com/onnx/models/blob/main/validated/vision/body_analysis/age_gender/levi_googlenet.py
function run(srcMat) {
    if (!self.net) {
        console.error('Gender model not loaded.');
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
    const result = { gender: genderList[argMax(output.data32F)], "gender-source": output.data32F };

    output.delete();
    inputBlob.delete();
    matC3.delete();

    return result;
}

