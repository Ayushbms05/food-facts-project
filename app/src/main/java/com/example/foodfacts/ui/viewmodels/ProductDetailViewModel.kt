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
import com.example.foodfacts.integration.prediction.DeepLearningRepository
import com.example.foodfacts.integration.prediction.PredictionResponse
import kotlinx.coroutines.launch

sealed class ProductDetailUiState {
    object Loading : ProductDetailUiState()
    data class Success(val product: ProductDomainModel) : ProductDetailUiState()
    data class Error(val message: String, val cachedProduct: ProductDomainModel? = null) : ProductDetailUiState()
}

sealed class AiPredictionState {
    object Idle : AiPredictionState()
    object Loading : AiPredictionState()
    data class Success(val prediction: PredictionResponse) : AiPredictionState()
    data class Error(val message: String) : AiPredictionState()
}

class ProductDetailViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: ProductRepository
    private val dlRepository = DeepLearningRepository()

    init {
        val database = AppDatabase.getInstance(application)
        repository = ProductRepository(
            productDao = database.productDao(),
            scanEventDao = database.scanEventDao()
        )
    }

    private val _uiState = MutableLiveData<ProductDetailUiState>(ProductDetailUiState.Loading)
    val uiState: LiveData<ProductDetailUiState> = _uiState

    private val _aiPredictionState = MutableLiveData<AiPredictionState>(AiPredictionState.Idle)
    val aiPredictionState: LiveData<AiPredictionState> = _aiPredictionState

    fun loadProduct(barcode: String) {
        val cleanBarcode = barcode.trim()
        if (cleanBarcode.isEmpty()) {
            _uiState.value = ProductDetailUiState.Error("Invalid product barcode.")
            return
        }

        viewModelScope.launch {
            _uiState.value = ProductDetailUiState.Loading
            when (val result = repository.getProductByBarcode(cleanBarcode, ScanSource.MANUAL_SEARCH)) {
                is Resource.Success -> {
                    _uiState.value = ProductDetailUiState.Success(result.data)
                }
                is Resource.Error -> {
                    _uiState.value = ProductDetailUiState.Error(
                        message = result.message,
                        cachedProduct = result.cachedData
                    )
                }
                is Resource.Loading -> {
                    _uiState.value = ProductDetailUiState.Loading
                }
            }
        }
    }

    fun predictNovaWithAi(ingredientsText: String?) {
        val text = ingredientsText.orEmpty().trim()
        if (text.isEmpty()) {
            _aiPredictionState.value = AiPredictionState.Error("No ingredients text available to analyze.")
            return
        }

        viewModelScope.launch {
            _aiPredictionState.value = AiPredictionState.Loading
            when (val result = dlRepository.predictNovaGroup(text)) {
                is Resource.Success -> {
                    _aiPredictionState.value = AiPredictionState.Success(result.data)
                }
                is Resource.Error -> {
                    _aiPredictionState.value = AiPredictionState.Error(result.message)
                }
                is Resource.Loading -> {
                    _aiPredictionState.value = AiPredictionState.Loading
                }
            }
        }
    }
}
