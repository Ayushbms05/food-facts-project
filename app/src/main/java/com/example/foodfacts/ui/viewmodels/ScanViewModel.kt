package com.example.foodfacts.ui.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.example.foodfacts.data.local.AppDatabase
import com.example.foodfacts.data.model.ProductDomainModel
import com.example.foodfacts.data.model.Resource
import com.example.foodfacts.data.model.ScanSource
import com.example.foodfacts.data.repository.ProductRepository
import kotlinx.coroutines.launch

sealed class ScanState {
    object Idle : ScanState()
    object Loading : ScanState()
    data class Success(val product: ProductDomainModel) : ScanState()
    data class Error(val message: String, val cachedData: ProductDomainModel? = null) : ScanState()
}

class ScanViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: ProductRepository

    init {
        val database = AppDatabase.getInstance(application)
        repository = ProductRepository(
            productDao = database.productDao(),
            scanEventDao = database.scanEventDao()
        )
    }

    private val _scanState = MutableLiveData<ScanState>(ScanState.Idle)
    val scanState: LiveData<ScanState> = _scanState

    private var lastScannedBarcode: String? = null
    private var lastScanTime: Long = 0

    fun onBarcodeScanned(rawBarcode: String, source: ScanSource = ScanSource.BARCODE) {
        val cleanBarcode = rawBarcode.trim()
        if (cleanBarcode.isEmpty()) return

        val now = System.currentTimeMillis()
        if (cleanBarcode == lastScannedBarcode && (now - lastScanTime) < 3000) {
            return // Debounce duplicate rapid scans
        }
        lastScannedBarcode = cleanBarcode
        lastScanTime = now

        fetchProductByBarcode(cleanBarcode, source)
    }

    fun fetchProductByBarcode(barcode: String, source: ScanSource = ScanSource.BARCODE) {
        viewModelScope.launch {
            _scanState.value = ScanState.Loading
            when (val result = repository.getProductByBarcode(barcode, source)) {
                is Resource.Success -> {
                    _scanState.value = ScanState.Success(result.data)
                }
                is Resource.Error -> {
                    _scanState.value = ScanState.Error(
                        message = result.message,
                        cachedData = result.cachedData
                    )
                }
                is Resource.Loading -> {
                    _scanState.value = ScanState.Loading
                }
            }
        }
    }

    fun resetState() {
        _scanState.value = ScanState.Idle
        lastScannedBarcode = null
    }
}
