let videoDuration = 0;
let isDragging = false;
let currentHandle = null;
let startTime = 0;
let endTime = 0;
let currentVideoUrl = null;

// Format time in seconds to MM:SS format
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Update timeline handles and labels
function updateTimeline() {
    const startHandle = document.getElementById('startHandle');
    const endHandle = document.getElementById('endHandle');
    const startTimeLabel = document.getElementById('startTimeLabel');
    const endTimeLabel = document.getElementById('endTimeLabel');
    const startTimeInput = document.getElementById('startTime');
    const endTimeInput = document.getElementById('endTime');
    const timelineSelection = document.getElementById('timelineSelection');

    // Ensure start time is not negative and less than end time
    startTime = Math.max(0, Math.min(startTime, endTime - 0.1));
    // Ensure end time is not greater than video duration and greater than start time
    endTime = Math.min(videoDuration, Math.max(startTime + 0.1, endTime));

    const startPercent = (startTime / videoDuration) * 100;
    const endPercent = (endTime / videoDuration) * 100;

    startHandle.style.left = `${startPercent}%`;
    endHandle.style.left = `${endPercent}%`;
    
    timelineSelection.style.left = `${startPercent}%`;
    timelineSelection.style.width = `${endPercent - startPercent}%`;
    
    startTimeLabel.textContent = formatTime(startTime);
    endTimeLabel.textContent = formatTime(endTime);
    
    // Update input values without triggering input event
    startTimeInput.value = startTime.toFixed(1);
    endTimeInput.value = endTime.toFixed(1);

    // Update video current time
    const video = document.getElementById('uploadedVideo');
    if (video.currentTime < startTime) {
        video.currentTime = startTime;
    } else if (video.currentTime > endTime) {
        video.currentTime = endTime;
    }
}

// Handle timeline interactions
document.getElementById('videoTimeline').addEventListener('mousedown', function(e) {
    const timeline = document.getElementById('videoTimeline');
    const rect = timeline.getBoundingClientRect();
    const clickPosition = (e.clientX - rect.left) / rect.width;
    const time = clickPosition * videoDuration;

    // Determine which handle is closer
    const startHandle = document.getElementById('startHandle');
    const endHandle = document.getElementById('endHandle');
    const startRect = startHandle.getBoundingClientRect();
    const endRect = endHandle.getBoundingClientRect();

    if (Math.abs(e.clientX - startRect.left) < Math.abs(e.clientX - endRect.left)) {
        currentHandle = 'start';
        startTime = Math.max(0, Math.min(time, endTime - 0.1));
    } else {
        currentHandle = 'end';
        endTime = Math.max(startTime + 0.1, Math.min(time, videoDuration));
    }

    isDragging = true;
    updateTimeline();
});

document.addEventListener('mousemove', function(e) {
    if (!isDragging) return;

    const timeline = document.getElementById('videoTimeline');
    const rect = timeline.getBoundingClientRect();
    const position = (e.clientX - rect.left) / rect.width;
    const time = position * videoDuration;

    if (currentHandle === 'start') {
        startTime = Math.max(0, Math.min(time, endTime - 0.1));
    } else {
        endTime = Math.max(startTime + 0.1, Math.min(time, videoDuration));
    }

    updateTimeline();
});

document.addEventListener('mouseup', function() {
    isDragging = false;
    currentHandle = null;
});

// Handle video upload
document.getElementById('videoInput').addEventListener('change', async function(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        
        if (result.error) {
            alert(result.error);
            return;
        }

        const video = document.getElementById('uploadedVideo');
        video.src = result.video_url;
        currentVideoUrl = result.video_url;
        
        video.onloadedmetadata = function() {
            videoDuration = video.duration;
            endTime = videoDuration;
            updateTimeline();
        };

        // Add video timeupdate event listener
        video.addEventListener('timeupdate', function() {
            const progress = (video.currentTime / videoDuration) * 100;
            document.getElementById('timelineProgress').style.width = `${progress}%`;
        });

    } catch (error) {
        console.error('Error uploading video:', error);
        alert('Error uploading video. Please try again.');
    }
});

// Handle video cutting
document.getElementById('cutButton').addEventListener('click', async function() {
    if (!currentVideoUrl) {
        alert('Please upload a video first');
        return;
    }

    const loading = document.getElementById('loading');
    const cutButton = document.getElementById('cutButton');
    
    loading.style.display = 'block';
    loading.textContent = 'Processing video, please wait...';
    cutButton.disabled = true;

    try {
        const response = await fetch('/cut', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                startTime: startTime,
                endTime: endTime,
                inputVideo: currentVideoUrl
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }

        // Display cut video in input video player
        const inputVideo = document.getElementById('inputVideo');
        
        // Clear any existing source
        inputVideo.removeAttribute('src');
        inputVideo.load();
        
        // Set new source
        const cutVideoUrl = result.video_url.startsWith('/') ? result.video_url : '/' + result.video_url;
        console.log('Setting video source to:', cutVideoUrl); // Debug log
        
        // Add event listeners before setting src
        inputVideo.onloadeddata = function() {
            console.log('Cut video loaded successfully');
            loading.style.display = 'none';
            cutButton.disabled = false;
        };
        
        inputVideo.onerror = function(e) {
            console.error('Error loading cut video:', e);
            console.error('Video error code:', inputVideo.error.code);
            console.error('Video error message:', inputVideo.error.message);
            alert('Error loading cut video. Please try again.');
            loading.style.display = 'none';
            cutButton.disabled = false;
        };

        // Set the source after adding event listeners
        inputVideo.src = cutVideoUrl;

    } catch (error) {
        console.error('Error cutting video:', error);
        alert('Error cutting video. Please try again.');
        loading.style.display = 'none';
        cutButton.disabled = false;
    }
});

// Handle video processing
async function processVideo() {
    const inputVideo = document.getElementById('inputVideo');
    if (!inputVideo.src) {
        alert('Please cut a video segment first');
        return;
    }

    const loading = document.getElementById('loading');
    const processButton = document.getElementById('processButton');
    
    loading.style.display = 'block';
    processButton.disabled = true;

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                inputVideo: inputVideo.src
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }

        // Display processed video
        document.getElementById('outputVideo').src = result.video_url;

        // Display bounce images
        const bounceContainer = document.getElementById('bounceImagesContainer');
        bounceContainer.innerHTML = '<h2>Bounce Points</h2>';
        
        if (result.bounce_frames && result.bounce_frames.length > 0) {
            result.bounce_frames.forEach(bounce => {
                const bounceDiv = document.createElement('div');
                bounceDiv.className = 'bounce-image';
                bounceDiv.innerHTML = `
                    <p>Time: ${formatTime(bounce.time)} | <span style="color:${bounce.inout === 'IN' ? '#4CAF50' : '#e53935'}">${bounce.inout}</span></p>
                    <div style="display: flex; gap: 20px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">
                        <div>
                            <div style='font-size:13px;color:#333;margin-bottom:3px;'>Ball Zoom</div>
                            <img src="data:image/jpeg;base64,${bounce.image}" alt="Bounce point" style="max-width: 480px; max-height: 480px;">
                        </div>
                        ${bounce.minimap ? `<div><div style='font-size:13px;color:#333;margin-bottom:3px;'>2D Minimap (Zoom)</div><img src="data:image/jpeg;base64,${bounce.minimap}" alt="Minimap" style="max-width: 400px; max-height: 400px;"></div>` : ''}
                    </div>
                `;
                bounceContainer.appendChild(bounceDiv);
            });
        } else {
            bounceContainer.innerHTML += '<p>No bounce points detected</p>';
        }

    } catch (error) {
        console.error('Error processing video:', error);
        alert('Error processing video. Please try again.');
    } finally {
        loading.style.display = 'none';
        processButton.disabled = false;
    }
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    const video = document.getElementById('uploadedVideo');
    if (!video) return;

    switch(e.key) {
        case '[': // Set start time
            startTime = video.currentTime;
            updateTimeline();
            break;
        case ']': // Set end time
            endTime = video.currentTime;
            updateTimeline();
            break;
        case ' ': // Play/Pause
            e.preventDefault();
            if (video.paused) {
                video.play();
            } else {
                video.pause();
            }
            break;
    }
});

// Add process button click handler
document.getElementById('processButton').addEventListener('click', processVideo);

// Add input event listeners for time inputs
document.getElementById('startTime').addEventListener('input', function(e) {
    const newStartTime = parseFloat(e.target.value);
    if (!isNaN(newStartTime) && newStartTime >= 0 && newStartTime < endTime) {
        startTime = newStartTime;
        updateTimeline();
    }
});

document.getElementById('endTime').addEventListener('input', function(e) {
    const newEndTime = parseFloat(e.target.value);
    if (!isNaN(newEndTime) && newEndTime > startTime && newEndTime <= videoDuration) {
        endTime = newEndTime;
        updateTimeline();
    }
}); 