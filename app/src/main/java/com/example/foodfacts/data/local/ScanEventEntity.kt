package com.example.foodfacts.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.example.foodfacts.data.model.ScanEvent
import com.example.foodfacts.data.model.ScanSource

@Entity(tableName = "scan_events")
data class ScanEventEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val barcode: String,
    val normalizedBarcode: String,
    val productName: String,
    val brand: String,
    val productFound: Boolean,
    val novaGroup: Int?,
    val novaSource: String,
    val scanSource: String,
    val eventSource: String,
    val lookupStatus: String,
    val timestamp: Long
)

fun ScanEventEntity.toDomainModel(): ScanEvent {
    val source = try {
        ScanSource.valueOf(scanSource)
    } catch (_: Exception) {
        ScanSource.BARCODE
    }
    return ScanEvent(
        id = id,
        barcode = barcode,
        normalizedBarcode = normalizedBarcode,
        productName = productName,
        brand = brand,
        productFound = productFound,
        novaGroup = novaGroup,
        novaSource = novaSource,
        scanSource = source,
        eventSource = eventSource,
        lookupStatus = lookupStatus,
        timestamp = timestamp
    )
}

fun ScanEvent.toEntity(): ScanEventEntity {
    return ScanEventEntity(
        id = id,
        barcode = barcode,
        normalizedBarcode = normalizedBarcode,
        productName = productName,
        brand = brand,
        productFound = productFound,
        novaGroup = novaGroup,
        novaSource = novaSource,
        scanSource = scanSource.name,
        eventSource = eventSource,
        lookupStatus = lookupStatus,
        timestamp = timestamp
    )
}
