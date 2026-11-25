package com.example.foodfacts.ui.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.foodfacts.data.local.AppDatabase
import com.example.foodfacts.data.model.ProductDomainModel
import com.example.foodfacts.data.model.ScanEvent
import com.example.foodfacts.data.repository.ProductRepository
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class HistoryViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: ProductRepository

    init {
        val database = AppDatabase.getInstance(application)
        repository = ProductRepository(
            productDao = database.productDao(),
            scanEventDao = database.scanEventDao()
        )
    }

    val historyProducts: StateFlow<List<ProductDomainModel>> =
        repository.getHistoryProductsFlow().stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )

    val scanEvents: StateFlow<List<ScanEvent>> =
        repository.getScanEventsFlow().stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )

    fun deleteHistoryProduct(barcode: String) {
        viewModelScope.launch {
            repository.deleteHistoryProduct(barcode)
        }
    }

    fun clearAllHistory() {
        viewModelScope.launch {
            repository.clearHistory()
        }
    }
}
