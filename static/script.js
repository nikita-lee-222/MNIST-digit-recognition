const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

//Background settings
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.strokeStyle = "white";
ctx.lineWidth = 15;

let drawing = false;

canvas.addEventListener("mousedown", (e) => {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY); 
});

//Exceptions
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseout", () => drawing = false);

canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
});

//Cleaning after task
function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

//We are sending image to the server for prediction
async function predict() {
    const dataUrl = canvas.toDataURL("image/png");
    const response = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({image: dataUrl})
    });
    const result = await response.json();
    if(result.error){
        document.getElementById("result").innerText = "Error: " + result.error;
    } else {
        document.getElementById("result").innerText =
            `Model said: ${result.digit} (Accuracy: ${result.confidence}%)`;
    }
}
