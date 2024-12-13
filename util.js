/**
 * urlで指定されたonnxモデルをロードします.
 * opencvはロードされている前提の処理です.
 * @param url
 * @returns {Promise<*>}
 */
export const loadModel = async function (url) {
    const response = await fetch(url, {method: 'GET', mode:"cors"});
    const buffer = await response.arrayBuffer();
    const data = new Uint8Array(buffer);
    const path = url.split('/').pop().split('?')[0];
    cv.FS_createDataFile('/', path, data, true, false, false);
    document.getElementById('status').innerText += `${path} Loaded!\n`;
    return cv.readNetFromONNX(path);
};
