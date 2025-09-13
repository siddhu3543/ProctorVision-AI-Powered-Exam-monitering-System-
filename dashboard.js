// static/dashboard.js
const socket = io.connect("http://localhost:5000");
const logBox = document.querySelector("#alerts");

socket.on("alert", (msg) => {
    let p = document.createElement("p");
    p.innerText = msg;
    logBox.appendChild(p);
});
