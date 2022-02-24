---
layout: post
title: 'ECC for Image Encryption'
date: '2022-02-02 15:04'
excerpt: >-
  Write an ECC implementaion for deep image steganongraphy
comments: true
tags: [february_2022, crypto]
---

### Finding an ECC Implementation:

- Try to follow the bitcoin blog by andrej kaparthy
- Take the public key of the reciver as the one where the secret key is 1 for convinience
- Step 4 is done now

### Step 5

**Point addition of keyPb for each Pm value and store it as cipher text Pc**

- Mentioned 3G as public key in ecc.py and used it in encrypt.py
- Hardcoded key_length to be 256 in ecc.py
