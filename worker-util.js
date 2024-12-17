const cache = {};

/**
 * urlで指定されたonnxモデルをロードします.
 * opencvはロードされている前提の処理です.
 * @param url
 * @returns {Promise<*>}
 */
const loadModel = function (cv, url) {

    const path = url.split('/').pop().split('?')[0];

    if (cache[path]) {
        console.log("load from cache", path);
        return Promise.resolve(cache[path]);
    }

    return fetchModel(url)
        .then(data => {
            saveModel(path, data)
            const model = cv.readNetFromONNX(path);
            cache[path] = model;
            console.log(path, printStringVector(model.getUnconnectedOutLayersNames()));
            return model;
        }).catch(e => console.error(e));
};

const fetchModel = (url) => {
    return fetch(url, { method: 'GET', mode: "cors" })
    .then((response) => response.arrayBuffer())
    .then((buffer) => new Uint8Array(buffer))
    .catch((e) => console.error(e));
}

const saveModel = async (name, data) => {
    cv.FS_createDataFile('/', name, data, true, false, false);
}

const printStringVector = (sv) => {
    let arr = new Array(sv.size() || 0);
    for (let i = 0; i < sv.size(); i++) {
        arr.push(sv.get(i));
    }
    return arr;
}


function argMax(data) {
    return data.reduce((acc, currentValue, currentIndex, arr) => arr[acc] < arr[currentIndex] ? currentIndex : acc, 0);
}

function adjustMatStep(mat, newStep) {
    const rows = mat.rows;
    const cols = mat.cols;
    const type = mat.type();

    // Create a new Mat with the desired step
    const newMat = new cv.Mat(rows, cols, type);
    const bytesPerPixel = mat.elemSize();

    for (let i = 0; i < rows; i++) {
        const srcOffset = i * mat.step;
        const dstOffset = i * newStep;

        for (let j = 0; j < cols * bytesPerPixel; j++) {
            newMat.data[dstOffset + j] = mat.data[srcOffset + j];
        }
    }

    return newMat;
}