
async function getMediaStream(constraints) {
  try {
    mediaStream =  await navigator.mediaDevices.getUserMedia(constraints);
    let video = document.getElementById('cam');
    video.srcObject = mediaStream;
    video.onloadedmetadata = (event) => {
      video.play();
    };
  } catch (err)  {
    console.error(err.message);
  }
  return mediaStream;
}

async function stopCamera(mediaStream) {
  try {
    // stop the current video stream
    if (mediaStream != null && mediaStream.active) {
      const tracks = mediaStream.getVideoTracks();
      tracks.forEach(track => {
        track.stop();
      })
    }

    // set the video source to null
    document.getElementById('cam').srcObject = null;
  } catch (err)  {
    console.error(err.message);
    alert(err.message);
  }
}

async function switchCamera(constraints) {
  try {
    // get new media stream
    mediaStream = await getMediaStream(constraints);
  } catch (err)  {
    console.error(err.message);
    alert(err.message);
  }

  return mediaStream;
}
