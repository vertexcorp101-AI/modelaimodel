# Build tiny-Vertex APK on Colab

Colab has Android SDK + NDK + fast CPU. Build time ~10-15 min.

## Colab cells (in order)

**1. Install Android SDK + NDK + JDK**
```
!apt-get update -qq && apt-get install -y -qq openjdk-17-jdk cmake
!wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -O /tmp/cmd.zip
!unzip -q /tmp/cmd.zip -d /tmp/cmdtools
!mkdir -p /opt/android-sdk/cmdline-tools/latest
!mv /tmp/cmdtools/cmdline-tools/* /opt/android-sdk/cmdline-tools/latest/
!rm -rf /tmp/cmdtools
!export ANDROID_HOME=/opt/android-sdk
!export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin
!yes | sdkmanager --licenses > /dev/null
!sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0" "ndk;27.1.12297006" > /dev/null
!echo "SDK ready: $(ls $ANDROID_HOME/platforms)"
```

**2. Clone project + model**
```
from google.colab import drive
drive.mount('/content/drive')
!rm -rf /content/apk-build
!git clone https://github.com/vertexcorp101-AI/modelaimodel /content/apk-build
!mkdir -p /content/apk-build/app/src/main/assets
!cp "/content/drive/MyDrive/tinyvertex-360m-Q4_K_M.gguf" /content/apk-build/app/src/main/assets/
!ls -la /content/apk-build/app/src/main/assets/
```

**3. Clone llama.cpp (pinned tag)**
```
!git clone --depth 1 --branch b7123 https://github.com/ggml-org/llama.cpp /content/apk-build/app/src/main/cpp/llama.cpp
```

**4. Build APK**
```
import os
os.environ["ANDROID_HOME"] = "/opt/android-sdk"
os.environ["ANDROID_NDK_HOME"] = "/opt/android-sdk/ndk/27.1.12297006"
os.environ["PATH"] = os.environ["ANDROID_HOME"] + "/platform-tools:" + os.environ["ANDROID_HOME"] + "/cmdline-tools/latest/bin:" + os.environ["PATH"]
os.chdir("/content/apk-build")
!chmod +x gradlew
!./gradlew assembleRelease --no-daemon --max-workers=2
```

**5. Save to Drive**
```
!mkdir -p /content/drive/MyDrive/apk
!cp /content/apk-build/app/build/outputs/apk/release/app-release.apk /content/drive/MyDrive/apk/tinyvertex-v1.apk
!ls -la /content/drive/MyDrive/apk/
```

## Notes
- First build downloads Gradle + dependencies (~5 min). Subsequent builds are faster.
- APK size: ~280-300MB (model 271MB + native libs ~15MB + app ~10MB).
- The `assembleRelease` task compiles llama.cpp AArch64 via CMake; takes ~3-5 min on Colab's 2-core CPU.
- If build fails with `ninja` errors, check `ANDROID_NDK_HOME` is set correctly.
