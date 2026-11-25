package com.example.foodfacts.util

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import java.util.Locale
import kotlin.coroutines.resume

object OcrHelper {

    private val recognizer by lazy {
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    }

    private val commonAllergens = listOf(
        "milk", "dairy", "egg", "eggs", "peanut", "peanuts",
        "tree nut", "nuts", "almond", "walnut", "cashew", "hazelnut", "pistachio",
        "wheat", "gluten", "soy", "soya", "soybean", "fish", "crustacean",
        "shellfish", "shrimp", "prawn", "crab", "lobster", "sesame", "mustard",
        "sulfite", "sulfites", "celery", "mollusc", "lupin"
    )

    suspend fun processImageUri(context: Context, uri: Uri): String = suspendCancellableCoroutine { continuation ->
        try {
            val inputImage = InputImage.fromFilePath(context, uri)
            recognizer.process(inputImage)
                .addOnSuccessListener { text ->
                    continuation.resume(text.text)
                }
                .addOnFailureListener { e ->
                    continuation.resume("Error reading text from image: ${e.localizedMessage}")
                }
        } catch (e: Exception) {
            continuation.resume("Failed to process image file: ${e.localizedMessage}")
        }
    }

    suspend fun processBitmap(bitmap: Bitmap): String = suspendCancellableCoroutine { continuation ->
        try {
            val inputImage = InputImage.fromBitmap(bitmap, 0)
            recognizer.process(inputImage)
                .addOnSuccessListener { text ->
                    continuation.resume(text.text)
                }
                .addOnFailureListener { e ->
                    continuation.resume("Error reading text from bitmap: ${e.localizedMessage}")
                }
        } catch (e: Exception) {
            continuation.resume("Failed to process bitmap: ${e.localizedMessage}")
        }
    }

    fun detectAllergens(text: String): List<String> {
        val lowerText = text.lowercase(Locale.ROOT)
        return commonAllergens.filter { allergen ->
            lowerText.contains(allergen)
        }.distinct().map { allergen ->
            allergen.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.ROOT) else it.toString() }
        }
    }
}
