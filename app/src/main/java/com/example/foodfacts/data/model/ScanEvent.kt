package com.example.foodfacts.data.model

enum class ScanSource {
    BARCODE,
    OCR,
    MANUAL_SEARCH
}

data class ScanEvent(
    val id: Long = 0,
    val barcode: String,
    val normalizedBarcode: String = barcode.padStart(14, '0'),
    val productName: String,
    val brand: String,
    val productFound: Boolean = true,
    val novaGroup: Int? = null,
    val novaSource: String = "OPEN_FOOD_FACTS", // OPEN_FOOD_FACTS, DL_MODEL, NONE
    val scanSource: ScanSource = ScanSource.BARCODE,
    val eventSource: String = "ANDROID_APP",
    val lookupStatus: String = "KNOWN_WITH_NOVA", // KNOWN_WITH_NOVA, KNOWN_WITHOUT_NOVA, UNKNOWN_PRODUCT, NETWORK_FAILURE
    val timestamp: Long = System.currentTimeMillis()
)
