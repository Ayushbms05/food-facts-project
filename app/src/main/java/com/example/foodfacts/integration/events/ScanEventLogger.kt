package com.example.foodfacts.integration.events

import com.example.foodfacts.data.local.ScanEventDao
import com.example.foodfacts.data.local.toEntity
import com.example.foodfacts.data.model.ProductDomainModel
import com.example.foodfacts.data.model.ScanEvent
import com.example.foodfacts.data.model.ScanSource

class ScanEventLogger(
    private val scanEventDao: ScanEventDao
) {
    suspend fun logProductLookupEvent(
        barcode: String,
        scanSource: ScanSource,
        product: ProductDomainModel?,
        isNetworkError: Boolean
    ) {
        val cleanBarcode = barcode.trim()
        val normalizedBarcode = cleanBarcode.padStart(14, '0')

        val lookupStatus = when {
            isNetworkError -> "NETWORK_FAILURE"
            product == null -> "UNKNOWN_PRODUCT"
            product.novaGroup != null -> "KNOWN_WITH_NOVA"
            else -> "KNOWN_WITHOUT_NOVA"
        }

        val event = ScanEvent(
            barcode = cleanBarcode,
            normalizedBarcode = normalizedBarcode,
            productName = product?.name ?: "Unknown Product",
            brand = product?.brand ?: "Unknown Brand",
            productFound = product != null,
            novaGroup = product?.novaGroup,
            novaSource = if (product?.novaGroup != null) "OPEN_FOOD_FACTS" else "NONE",
            scanSource = scanSource,
            eventSource = "ANDROID_APP",
            lookupStatus = lookupStatus,
            timestamp = System.currentTimeMillis()
        )

        scanEventDao.insertEvent(event.toEntity())
    }

    suspend fun logCustomEvent(event: ScanEvent) {
        scanEventDao.insertEvent(event.toEntity())
    }
}
