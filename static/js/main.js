const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const preview = document.getElementById('preview');
const captureBtn = document.getElementById('capture');
const switchBtn = document.getElementById('switchCamera');
const webcamForm = document.getElementById('webcamForm');
const webcamFile = document.getElementById('webcamFile');

let useBackCamera = false;
let currentStream;

async function startCamera() {
    if (currentStream) currentStream.getTracks().forEach(track => track.stop());
    const facingMode = useBackCamera ? { exact: "environment" } : "user";

    try {
        currentStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode } });
        video.srcObject = currentStream;
    } catch (err) {
        alert("Error accessing camera: " + err);
    }
}

switchBtn.addEventListener('click', () => {
    useBackCamera = !useBackCamera;
    startCamera();
});

captureBtn.addEventListener('click', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    // Show preview
    preview.src = canvas.toDataURL('image/jpeg');
    preview.style.display = 'block';

    // Convert canvas to file
    canvas.toBlob(blob => {
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        webcamFile.files = dataTransfer.files;

        // Prompt for Student ID
        const studentId = prompt("Enter Student ID:");
        if (!studentId) return alert("Student ID is required.");

        // Update hidden field and form action dynamically
        document.getElementById('webcamStudentId').value = studentId;
        webcamForm.action = `/attendance/capture/${studentId}`;
        webcamForm.style.display = 'block';
    }, 'image/jpeg');
});

// Reset after submission
webcamForm.addEventListener('submit', () => {
    setTimeout(() => {
        preview.src = "";
        preview.style.display = 'none';
        webcamForm.style.display = 'none';
        webcamFile.value = "";
        document.getElementById('webcamStudentId').value = "";
    }, 500);
});

// Fix for Upload Form 404 issue
const uploadForm = document.querySelector('form[action="/attendance/capture/"]');
if (uploadForm) {
    uploadForm.addEventListener('submit', function (e) {
        const studentId = document.getElementById('studentIdUpload').value.trim();
        if (!studentId) {
            e.preventDefault();
            alert("Please enter a valid Student ID before uploading.");
            return;
        }
        this.action = `/attendance/capture/${studentId}`;
    });
}

// Start camera on load
startCamera();

