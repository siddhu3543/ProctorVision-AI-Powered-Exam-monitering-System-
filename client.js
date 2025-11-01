// static/client.js
const video = document.querySelector("#videoElement");
const warningBox = document.querySelector("#warning");

const socket = io.connect("http://localhost:5000");

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Camera access denied:", err);
    });

// Send frames every second
setInterval(() => {
    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    let ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    socket.emit("frame", canvas.toDataURL("image/jpeg"));
}, 1000);

// Receive alerts from server
socket.on("alert", (msg) => {
    warningBox.innerText = "⚠️ Warning: " + msg;
    warningBox.style.display = "block";
});
