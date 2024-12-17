export function drawImage(ctx, img) {
    const width = img.width;
    const height = img.height;
    // Draw the image on the canvas
    ctx.drawImage(img, 0, 0, width, height);
}

export async function drawEmotionLabels(ctx, detections) {
    detections.forEach(det => {
        const {emotions, faceRectangle} = det;
        const {top, left} = faceRectangle;

        // Find the emotion with the highest value
        const maxEmotion = Object.keys(emotions).reduce((a, b) => emotions[a] > emotions[b] ? a : b);

        // Draw the label on the canvas
        ctx.fillStyle = 'red';
        ctx.font = '16px Arial';
        ctx.fillText(`${maxEmotion}: ${emotions[maxEmotion]}`, left, top - 5);
    });
}