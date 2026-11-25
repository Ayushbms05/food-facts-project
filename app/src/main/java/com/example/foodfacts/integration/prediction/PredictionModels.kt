package com.example.foodfacts.integration.prediction

import com.google.gson.annotations.SerializedName

data class PredictionRequest(
    @SerializedName("ingredients_text")
    val ingredientsText: String
)

data class PredictionResponse(
    @SerializedName("predicted_nova_group")
    val predictedNovaGroup: Int?,
    @SerializedName("confidence_score")
    val confidenceScore: Double?,
    @SerializedName("ingredient_evidence")
    val ingredientEvidence: List<String>?,
    @SerializedName("explanation")
    val explanation: String?,
    @SerializedName("model_version")
    val modelVersion: String? = "v1.0-dl-member2"
)
