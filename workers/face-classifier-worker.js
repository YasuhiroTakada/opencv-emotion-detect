workers = {};

dataHolder = [];

function start(workerName, index, context, args) {
    const workerId = `${workerName}Worker`;
    let worker = self.workers[workerId];
    if (!worker) {
        console.log("create worker", workerId);
        worker = new Worker(`./${workerName}-detect-worker.js`);
        worker.onmessage = complete;
        self.workers[workerId] = worker;
    }
    console.log(`start ${workerName} worker`, args, worker);
    worker.postMessage(args);
}

function complete(msg) {
    const [index, workerResult] = [...msg.data];
    console.log("worker complete", msg, dataHolder);
    const context = dataHolder[index] || {};
    const result = { ...context, ...workerResult }
    if (result.age && result.gender && result.emotions) {
        self.postMessage(result);
        delete dataHolder[index];
    } else {
        dataHolder[index] = result;
    }
}

/**
 * worker message listener function.
 * 0: image rows(height), 1: image cols(width), 2: image data(Uint8Array), 3: id, 4: run worker names(array[string]);, 5: classifier context.
 * @param {*} message
 */
self.onmessage = (message) => {
    console.log('classfier worker', message.data);
    const [rows, cols, bytes, index, workerNames, context] = [...message.data];
    dataHolder[index] = { ...context };
    workerNames.forEach(workerName => {
        start(workerName, index, context, [rows, cols, bytes, index]);
    });
}
