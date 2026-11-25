package com.example.foodfacts.ui.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.example.foodfacts.data.local.AppDatabase
import com.example.foodfacts.data.model.ComparisonResult
import com.example.foodfacts.data.model.ProductDomainModel
import com.example.foodfacts.data.model.Resource
import com.example.foodfacts.data.repository.ProductRepository
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

sealed class CompareUiState {
    object SelectionPending : CompareUiState()
    object Loading : CompareUiState()
    data class Success(val comparisonResult: ComparisonResult) : CompareUiState()
    data class Error(val message: String) : CompareUiState()
}

class CompareViewModel(application: Application) : AndroidViewModel(application) {

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

    private val _productA = MutableLiveData<ProductDomainModel?>()
    val productA: LiveData<ProductDomainModel?> = _productA

    private val _productB = MutableLiveData<ProductDomainModel?>()
    val productB: LiveData<ProductDomainModel?> = _productB

    private val _uiState = MutableLiveData<CompareUiState>(CompareUiState.SelectionPending)
    val uiState: LiveData<CompareUiState> = _uiState

    fun setProductA(product: ProductDomainModel) {
        _productA.value = product
        evaluateComparison()
    }

    fun setProductB(product: ProductDomainModel) {
        _productB.value = product
        evaluateComparison()
    }

    fun loadAndSetProductA(barcode: String) {
        viewModelScope.launch {
            _uiState.value = CompareUiState.Loading
            when (val res = repository.getProductByBarcode(barcode)) {
                is Resource.Success -> setProductA(res.data)
                is Resource.Error -> {
                    if (res.cachedData != null) setProductA(res.cachedData)
                    else _uiState.value = CompareUiState.Error("Product A error: ${res.message}")
                }
                else -> {}
            }
        }
    }

    fun loadAndSetProductB(barcode: String) {
        viewModelScope.launch {
            _uiState.value = CompareUiState.Loading
            when (val res = repository.getProductByBarcode(barcode)) {
                is Resource.Success -> setProductB(res.data)
                is Resource.Error -> {
                    if (res.cachedData != null) setProductB(res.cachedData)
                    else _uiState.value = CompareUiState.Error("Product B error: ${res.message}")
                }
                else -> {}
            }
        }
    }

    private fun evaluateComparison() {
        val pA = _productA.value
        val pB = _productB.value

        if (pA != null && pB != null) {
            val result = repository.compareProducts(pA, pB)
            _uiState.value = CompareUiState.Success(result)
        } else {
            _uiState.value = CompareUiState.SelectionPending
        }
    }

    fun resetComparison() {
        _productA.value = null
        _productB.value = null
        _uiState.value = CompareUiState.SelectionPending
    }
}
