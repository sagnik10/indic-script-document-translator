param(
    [string]$ApiKey = "",
    [string]$Model = "gpt-4o-mini",
    [int]$Port = 8503
)

$ErrorActionPreference = "Stop"

if (-not $ApiKey) {
    $ApiKey = $env:OPENAI_API_KEY
}

if (-not $ApiKey) {
    $secureKey = Read-Host -Prompt "OpenAI API key" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $ApiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

if (-not $ApiKey) {
    throw "OPENAI_API_KEY is required."
}

$env:OPENAI_API_KEY = $ApiKey
$env:OPENAI_TRANSLATION_MODEL = $Model
$env:OPENAI_EXTRACTION_MODEL = $Model
$env:OCR_ENABLE_INDICTRANS2_PROVIDER = "0"

.\.venv\Scripts\python.exe -m streamlit run .\pretrained_page_ocr\streamlit_app.py --server.port $Port
