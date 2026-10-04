import os, glob, shutil
matches = glob.glob('/content/apk-build/**/app-release*.apk', recursive=True)
print('Found:', matches)
if matches:
    dst = '/content/drive/MyDrive/apk/tinyvertex-v1.apk'
    os.makedirs('/content/drive/MyDrive/apk', exist_ok=True)
    shutil.copy2(matches[0], dst)
    print('Copied to:', dst)