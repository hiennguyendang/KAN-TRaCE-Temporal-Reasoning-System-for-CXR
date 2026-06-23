@echo off
setlocal enabledelayedexpansion

:: --- CẤU HÌNH ĐƯỜNG DẪN ---
set "REMOTE_BASE=dhint:CHEX-DATA/Mimic-CXR/files"
set "LOCAL_BASE=C:\Users\dhint\CHEX-DATA"
set "OUTPUT_DIR=C:\Users\dhint\CHEX-DATA\MIMIC-CXR"
set "MIMIC_METADATA_CSV=C:\Users\dhint\CHEX-DATA\MIMIC-CXR\mimic-cxr-2.0.0-metadata.csv"

:: --- RCLONE TUNING (HIGH THROUGHPUT) ---
set "RCLONE_TRANSFERS=12"
set "RCLONE_CHECKERS=32"
set "RCLONE_DRIVE_CHUNK_SIZE=256M"
set "RCLONE_BUFFER_SIZE=128M"
set "RCLONE_MULTI_THREAD_STREAMS=8"
set "RCLONE_MULTI_THREAD_CUTOFF=10M"
set "RCLONE_RETRIES=6"
set "RCLONE_LOW_LEVEL_RETRIES=20"
set "RCLONE_RETRIES_SLEEP=5s"
set "RCLONE_TIMEOUT=10m"
set "RCLONE_CONTIMEOUT=30s"
set "RCLONE_STATS=1s"
set "RCLONE_LOG_LEVEL=ERROR"
set "RCLONE_LOG_DIR=%OUTPUT_DIR%\logs"
set "BATCH_LOG=%RCLONE_LOG_DIR%\mimic_batch.log"

if not exist "%RCLONE_LOG_DIR%" (
    mkdir "%RCLONE_LOG_DIR%"
)

where rclone >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Khong tim thay rclone trong PATH.
    exit /b 1
)

if not exist "%MIMIC_METADATA_CSV%" (
    echo [ERROR] Khong tim thay file metadata CSV: %MIMIC_METADATA_CSV%
    exit /b 1
)

:: Vòng lặp chạy từ Pack 10 đến 19
for /L %%i in (10,1,19) do (
    set "PACK=p%%i"

    echo.
    echo [%TIME%] DANG TAI !PACK! bang rclone...
    rclone copy "%REMOTE_BASE%/!PACK!" "%LOCAL_BASE%/!PACK!" --create-empty-src-dirs --progress --stats %RCLONE_STATS% --stats-log-level NOTICE --log-level %RCLONE_LOG_LEVEL% --log-file "%RCLONE_LOG_DIR%\rclone_!PACK!.log" --transfers %RCLONE_TRANSFERS% --checkers %RCLONE_CHECKERS% --drive-chunk-size %RCLONE_DRIVE_CHUNK_SIZE% --buffer-size %RCLONE_BUFFER_SIZE% --multi-thread-streams %RCLONE_MULTI_THREAD_STREAMS% --multi-thread-cutoff %RCLONE_MULTI_THREAD_CUTOFF% --contimeout %RCLONE_CONTIMEOUT% --timeout %RCLONE_TIMEOUT% --retries %RCLONE_RETRIES% --low-level-retries %RCLONE_LOW_LEVEL_RETRIES% --retries-sleep %RCLONE_RETRIES_SLEEP% --drive-skip-gdocs --fast-list --drive-pacer-min-sleep 10ms
    if errorlevel 1 (
        echo [ERROR] Tai !PACK! that bai, bo qua pack nay.
        echo [%TIME%] [ERROR] Download failed for !PACK!>>"%BATCH_LOG%"
    ) else (
        echo [%TIME%] DANG XU LY !PACK! bang Python...
        echo [%TIME%] [INFO] Start preprocessing !PACK!>>"%BATCH_LOG%"

        :: Xu ly va xoa local p-folder sau khi thanh cong
        python C:\Users\dhint\CHEX-DATA\MyChex\preprocess\mimic_main.py --source-dir "%LOCAL_BASE%" --output-dir "%OUTPUT_DIR%" --metadata-csv "%MIMIC_METADATA_CSV%" --p-folders !PACK! --pack-name !PACK! --delete-p-folders
        if errorlevel 1 (
            echo [ERROR] Preprocess !PACK! that bai.
            echo [%TIME%] [ERROR] Preprocess failed for !PACK!>>"%BATCH_LOG%"
        ) else (
            echo [%TIME%] DA XU LY XONG !PACK!
            echo [%TIME%] [INFO] Finished preprocessing !PACK!>>"%BATCH_LOG%"
        )
    )
)

echo.
echo ============================================================
echo [DONE] TAT CA CAC GOI DA DUOC TAI VA XU LY XONG!
echo ============================================================
pause