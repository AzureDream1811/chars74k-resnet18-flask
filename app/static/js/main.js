// Global variables
let canvas, ctx;
let isDrawing = false;
let currentFile = null;
let lastX = 0;
let lastY = 0;

// Initialize when page loads
window.onload = function() {
    initializeCanvas();
    initializeFileUpload();
    initializeDragAndDrop();
};

/**
 * Initialize canvas for drawing
 */
function initializeCanvas() {
    canvas = document.getElementById('draw-canvas');
    ctx = canvas.getContext('2d');
    
    // Set canvas drawing properties
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000';
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouchStart);
    canvas.addEventListener('touchmove', handleTouchMove);
    canvas.addEventListener('touchend', stopDrawing);
}

/**
 * Initialize file upload functionality
 */
function initializeFileUpload() {
    const fileInput = document.getElementById('file-input');
    fileInput.addEventListener('change', handleFileSelect);
}

/**
 * Initialize drag and drop functionality
 */
function initializeDragAndDrop() {
    const uploadArea = document.getElementById('upload-area');
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
}

/**
 * Switch between tabs
 * @param {string} tabName - Name of the tab to switch to
 */
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');

    // Hide results when switching tabs
    document.getElementById('results').classList.remove('show');
}

/**
 * Get mouse position relative to canvas
 * @param {MouseEvent} e - Mouse event
 * @returns {Object} Object with x and y coordinates
 */
function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

/**
 * Start drawing on canvas
 * @param {MouseEvent} e - Mouse event
 */
function startDrawing(e) {
    isDrawing = true;
    const pos = getMousePos(e);
    lastX = pos.x;
    lastY = pos.y;
}

/**
 * Draw on canvas
 * @param {MouseEvent} e - Mouse event
 */
function draw(e) {
    if (!isDrawing) return;
    
    const pos = getMousePos(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    
    lastX = pos.x;
    lastY = pos.y;
}

/**
 * Stop drawing on canvas
 */
function stopDrawing() {
    isDrawing = false;
}

/**
 * Handle touch start event
 * @param {TouchEvent} e - Touch event
 */
function handleTouchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

/**
 * Handle touch move event
 * @param {TouchEvent} e - Touch event
 */
function handleTouchMove(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

/**
 * Clear the canvas
 */
function clearCanvas() {
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#000';
    document.getElementById('results').classList.remove('show');
}

/**
 * Handle file selection
 * @param {Event} e - Change event
 */
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        currentFile = file;
        const reader = new FileReader();
        reader.onload = function(event) {
            document.getElementById('preview-image').src = event.target.result;
            document.getElementById('preview-section').classList.add('show');
            document.getElementById('results').classList.remove('show');
        };
        reader.readAsDataURL(file);
    }
}

/**
 * Handle drag over event
 * @param {DragEvent} e - Drag event
 */
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

/**
 * Handle drag leave event
 * @param {DragEvent} e - Drag event
 */
function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

/**
 * Handle drop event
 * @param {DragEvent} e - Drop event
 */
function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        currentFile = file;
        const reader = new FileReader();
        reader.onload = function(event) {
            document.getElementById('preview-image').src = event.target.result;
            document.getElementById('preview-section').classList.add('show');
            document.getElementById('results').classList.remove('show');
        };
        reader.readAsDataURL(file);
    }
}

/**
 * Reset upload section
 */
function resetUpload() {
    document.getElementById('preview-section').classList.remove('show');
    document.getElementById('results').classList.remove('show');
    document.getElementById('file-input').value = '';
    currentFile = null;
}

/**
 * Predict from uploaded image
 */
async function predictUpload() {
    if (!currentFile) return;

    const formData = new FormData();
    formData.append('image', currentFile);

    await sendPrediction(formData);
}

/**
 * Predict from drawn image
 */
async function predictDraw() {
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('image', blob, 'drawing.png');
        await sendPrediction(formData);
    });
}

/**
 * Send prediction request to server
 * @param {FormData} formData - Form data containing the image
 */
async function sendPrediction(formData) {
    // Show loading indicator
    document.getElementById('loading').classList.add('show');
    document.getElementById('results').classList.remove('show');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        displayResults(data);
    } catch (error) {
        alert('Prediction failed: ' + error.message);
    } finally {
        document.getElementById('loading').classList.remove('show');
    }
}

/**
 * Display prediction results
 * @param {Object} data - Response data from server
 */
function displayResults(data) {
    // Display top prediction
    document.getElementById('top-prediction').textContent = data.top_prediction;

    // Display top 3 results
    const top3Container = document.getElementById('top3-results');
    top3Container.innerHTML = '';

    data.top3_results.forEach((result, index) => {
        const card = document.createElement('div');
        card.className = 'top3-card' + (index === 0 ? ' rank-1' : '');
        card.innerHTML = `
            <div class="rank-badge">#${index + 1}</div>
            <div class="top3-char">${result.char}</div>
            <div class="confidence">${result.confidence}</div>
        `;
        top3Container.appendChild(card);
    });

    document.getElementById('results').classList.add('show');
}