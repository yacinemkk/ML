import os

COLAB_DATA_PATH = "/content/drive/MyDrive/PFE/IPFIX_ML_Instances/"
COLAB_OUTPUT_PATH = "/content/drive/MyDrive/PFE/results/"


def setup_colab_paths():
    from google.colab import drive

    drive.mount("/content/drive")
    os.makedirs(COLAB_OUTPUT_PATH, exist_ok=True)
    return COLAB_DATA_PATH, COLAB_OUTPUT_PATH


DATA_PATH = COLAB_DATA_PATH
OUTPUT_PATH = COLAB_OUTPUT_PATH

SDN_FEATURES = [
    "duration",
    "ipProto",
    "outPacketCount",
    "outByteCount",
    "inPacketCount",
    "inByteCount",
    "outSmallPktCount",
    "outLargePktCount",
    "outNonEmptyPktCount",
    "outDataByteCount",
    "outAvgIAT",
    "outFirstNonEmptyPktSize",
    "outMaxPktSize",
    "outStdevPayloadSize",
    "outStdevIAT",
    "outAvgPacketSize",
    "inSmallPktCount",
    "inLargePktCount",
    "inNonEmptyPktCount",
    "inDataByteCount",
    "inAvgIAT",
    "inFirstNonEmptyPktSize",
    "inMaxPktSize",
    "inStdevPayloadSize",
    "inStdevIAT",
    "inAvgPacketSize",
    "http",
    "https",
    "smb",
    "dns",
    "ntp",
    "tcp",
    "udp",
    "ssdp",
    "lan",
    "wan",
]

TARGET = "name"

TARGET_CLASSES = [
    "eclear",
    "sleep",
    "esensor",
    "hub-plus",
    "humidifier",
    "home-unit",
    "inkjet-printer",
    "smart-wifi-plug-mini",
    "smart-power-strip",
    "echo-dot",
    "fire7-tablet",
    "google-nest-mini",
    "google-chromecast",
    "atom-cam",
    "kasa-camera-pro",
    "kasa-smart-led-lamp",
    "fire-tv-stick-4k",
    "qrio-hub",
]

DTYPE_DICT = {
    "duration": "float32",
    "ipProto": "int16",
    "outPacketCount": "int32",
    "outByteCount": "int64",
    "inPacketCount": "int32",
    "inByteCount": "int64",
    "outSmallPktCount": "int32",
    "outLargePktCount": "int32",
    "outNonEmptyPktCount": "int32",
    "outDataByteCount": "int64",
    "outAvgIAT": "float32",
    "outFirstNonEmptyPktSize": "int32",
    "outMaxPktSize": "int32",
    "outStdevPayloadSize": "float32",
    "outStdevIAT": "float32",
    "outAvgPacketSize": "float32",
    "inSmallPktCount": "int32",
    "inLargePktCount": "int32",
    "inNonEmptyPktCount": "int32",
    "inDataByteCount": "int64",
    "inAvgIAT": "float32",
    "inFirstNonEmptyPktSize": "int32",
    "inMaxPktSize": "int32",
    "inStdevPayloadSize": "float32",
    "inStdevIAT": "float32",
    "inAvgPacketSize": "float32",
    "http": "int8",
    "https": "int8",
    "smb": "int8",
    "dns": "int8",
    "ntp": "int8",
    "tcp": "int8",
    "udp": "int8",
    "ssdp": "int8",
    "lan": "int8",
    "wan": "int8",
    "device": "category",
    "name": "category",
}
