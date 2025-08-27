# Most likely false positives

These files (and eight others) are most likely false positives. I can't see a
recent for me to have a total of ten files with `TRIANGULATION`.

Currently the only validation I have done is the following.

First start a docker container with [Debian](https://www.debian.org):

Then run the following command to update and install tools:

```bash
apt update && apt install -y opentype-sanitizer
```

Check the files:

```bash
root@4aabaebeb4be:/data# ots-sanitize MesloLGMNerdFont-Regular.ttf clean_MesloLGMNerdFont-Regular.ttf
WARNING: gasp: Changed the version number to 1
File sanitized successfully!
root@4aabaebeb4be:/data# ots-sanitize 4e85bc9ebe07e0340c9c4fc2f6c38908.ttf clean_4e85bc9ebe07e0340c9c4fc2f6c38908.ttf
File sanitized successfully!
root@d4aabaebeb4be:/data# ls -l *ttf
-rw-r--r-- 1 root root  356840 Aug 27 09:28 4e85bc9ebe07e0340c9c4fc2f6c38908.ttf
-rw-r--r-- 1 root root 2786484 Aug 27 09:28 MesloLGMNerdFont-Regular.ttf
-rw-r--r-- 1 root root  364024 Aug 27 09:39 clean_4e85bc9ebe07e0340c9c4fc2f6c38908.ttf
-rw-r--r-- 1 root root 2794164 Aug 27 09:37 clean_MesloLGMNerdFont-Regular.ttf
```
