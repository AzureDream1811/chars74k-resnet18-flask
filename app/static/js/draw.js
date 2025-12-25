const canvas = document.getElementById("myCanvas");
const context = canvas.getContext("2d");

var isDrawing = false;
var startX;
var startY;

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

function startDrawing(event) {
  if (event.button !== 0) return; // Check if the left mouse button is pressed
  isDrawing = true;
  startX = event.clientX - canvas.offsetLeft;
  startY = event.clientY - canvas.offsetTop;
}

function draw(event) {
  if (!isDrawing) return;

  var x = event.clientX - canvas.offsetLeft;
  var y = event.clientY - canvas.offsetTop;

  context.beginPath();
  context.moveTo(startX, startY);
  context.lineTo(x, y);
  context.stroke();
  context.closePath();

  startX = x;
  startY = y;
}

function stopDrawing() {
  isDrawing = false;
}

document.getElementById("clearButton").addEventListener("click", function () {
  context.clearRect(0, 0, canvas.width, canvas.height);
});
