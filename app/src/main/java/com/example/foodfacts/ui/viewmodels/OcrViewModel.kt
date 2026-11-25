package com.example.foodfacts.ui.viewmodels

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.example.foodfacts.data.local.AppDatabase
import com.example.foodfacts.data.model.Resource
import com.example.foodfacts.data.model.ScanSource
import com.example.foodfacts.data.repository.ProductRepository
import com.example.foodfacts.integration.prediction.DeepLearningRepository
import com.example.foodfacts.integration.prediction.PredictionResponse
import com.example.foodfacts.util.OcrHelper
import kotlinx.coroutines.launch

sealed class OcrUiState {
    object Idle : OcrUiState()
    object Processing : OcrUiState()
    data class Success(
        val extractedText: String,
        val detectedAllergens: List<String>
    ) : OcrUiState()
    data class Error(val message: String) : OcrUiState()
}

class OcrViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: ProductRepository
    private val dlRepository = DeepLearningRepository()

    init {
        val database = AppDatabase.getInstance(application)
        repository = ProductRepository(
            productDao = database.productDao(),
            scanEventDao = database.scanEventDao()
        )
    }

    private val _uiState = MutableLiveData<OcrUiState>(OcrUiState.Idle)
    val uiState: LiveData<OcrUiState> = _uiState

    private val _aiPredictionState = MutableLiveData<AiPredictionState>(AiPredictionState.Idle)
    val aiPredictionState: LiveData<AiPredictionState> = _aiPredictionState

    fun processImageUri(uri: Uri) {
        viewModelScope.launch {
            _uiState.value = OcrUiState.Processing
            val text = OcrHelper.processImageUri(getApplication(), uri)
            if (text.startsWith("Error") || text.startsWith("Failed")) {
                _uiState.value = OcrUiState.Error(text)
            } else {
                val allergens = OcrHelper.detectAllergens(text)
                _uiState.value = OcrUiState.Success(text, allergens)
                repository.recordScanEvent(
                    barcode = "OCR-${System.currentTimeMillis()}",
                    productName = "OCR Extracted Text (${text.take(20)}...)",
                    brand = "Image Scan",
                    scanSource = ScanSource.OCR
                )
            }
        }
    }

    fun processBitmap(bitmap: Bitmap) {
        viewModelScope.launch {
            _uiState.value = OcrUiState.Processing
            val text = OcrHelper.processBitmap(bitmap)
            if (text.startsWith("Error") || text.startsWith("Failed")) {
                _uiState.value = OcrUiState.Error(text)
            } else {
                val allergens = OcrHelper.detectAllergens(text)
                _uiState.value = OcrUiState.Success(text, allergens)
                repository.recordScanEvent(
                    barcode = "OCR-${System.currentTimeMillis()}",
                    productName = "OCR Bitmap (${text.take(20)}...)",
                    brand = "Camera Capture",
                    scanSource = ScanSource.OCR
                )
            }
        }
    }

    fun predictNovaWithAi(ingredientsText: String) {
        val cleanText = ingredientsText.trim()
        if (cleanText.isEmpty()) {
            _aiPredictionState.value = AiPredictionState.Error("Ingredients text is empty. Enter or scan text first.")
            return
        }
        viewModelScope.launch {
            _aiPredictionState.value = AiPredictionState.Loading
            when (val result = dlRepository.predictNovaGroup(cleanText)) {
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

    fun reset() {
        _uiState.value = OcrUiState.Idle
        _aiPredictionState.value = AiPredictionState.Idle
    }
}
