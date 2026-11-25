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
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

sealed class SearchUiState {
    object Idle : SearchUiState()
    object Loading : SearchUiState()
    data class Success(val products: List<ProductDomainModel>) : SearchUiState()
    object Empty : SearchUiState()
    data class Error(val message: String) : SearchUiState()
}

class SearchViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: ProductRepository

    init {
        val database = AppDatabase.getInstance(application)
        repository = ProductRepository(
            productDao = database.productDao(),
            scanEventDao = database.scanEventDao()
        )
    }

    private val _uiState = MutableLiveData<SearchUiState>(SearchUiState.Idle)
    val uiState: LiveData<SearchUiState> = _uiState

    private var searchJob: Job? = null

    fun search(query: String, debounceMs: Long = 400) {
        searchJob?.cancel()
        val cleanQuery = query.trim()
        if (cleanQuery.isEmpty()) {
            _uiState.value = SearchUiState.Idle
            return
        }

        searchJob = viewModelScope.launch {
            if (debounceMs > 0) {
                delay(debounceMs)
            }
            _uiState.value = SearchUiState.Loading
            when (val result = repository.searchProducts(cleanQuery)) {
                is Resource.Success -> {
                    if (result.data.isEmpty()) {
                        _uiState.value = SearchUiState.Empty
                    } else {
                        _uiState.value = SearchUiState.Success(result.data)
                    }
                }
                is Resource.Error -> {
                    _uiState.value = SearchUiState.Error(result.message)
                }
                is Resource.Loading -> {
                    _uiState.value = SearchUiState.Loading
                }
            }
        }
    }

    fun recordManualSelectEvent(product: ProductDomainModel) {
        viewModelScope.launch {
            repository.recordScanEvent(
                barcode = product.barcode,
                productName = product.name,
                brand = product.brand,
                scanSource = ScanSource.MANUAL_SEARCH
            )
        }
    }
}
