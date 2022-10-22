---
layout: post
title: 'Hacking together yt-whisper-subtitle pipeline'
date: '2022-10-22 03:15'
excerpt: >-
  Using OpenAI's whisper to generate subtitle embeded yt-video
comments: true
tags: [october_2022, hacking]
---

**UPD** : Final fork available at : [whisper.cpp](https://github.com/ChanBong/whisper.cpp)

## Objectives 

- Transcribe a part, or full you tube video 
- Least amount of downloading 
- Translation, if required 
- CLI

-----

**11:21:50** : Found [this](https://gist.github.com/DaniruKun/96f763ec1a037cc92fe1a059b643b818). Claims to download the video, transcribe and translate it
- Seems legit.
- TO make it work for a portion of input
    - Avoid full download, i.e, only download the required portion


**11:42:33** : Apparantely if you give ffmpeg a video source it can download only the specified portion
- [here](https://unix.stackexchange.com/questions/230481/how-to-download-portion-of-video-with-youtube-dl-command)


**11:52:58** : Need some bash scripting to 
- Get both audio and video stream seprately from yt-dlp
- Use -1 for downloading full video


**11:59:34** : Forking whisper.cpp is the best idea 
- Setup small model 
- keep results in res folder 
- WO : Getting with a single res.mp4


**12:29:34** : Final Pipeline
- Give a youtube url with starting point and duration you want to clip 
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) fetches video and audio streams. 
- [ffmpeg](https://ffmpeg.org) downloads part of the video
- [whisper](https://github.com/ggerganov/whisper.cpp) cli tool transcribes and if needed translates to english and store it as .srt file
- Use ffmped to embed this srt file to the video stream
- cleanup the residual files 

-----
