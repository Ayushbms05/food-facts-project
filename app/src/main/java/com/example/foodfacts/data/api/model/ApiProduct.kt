package com.example.foodfacts.data.api.model

import com.google.gson.annotations.SerializedName

data class ApiProduct(
    @SerializedName("code")
    val code: String? = null,
    @SerializedName("product_name")
    val productName: String? = null,
    @SerializedName("brands")
    val brands: String? = null,
    @SerializedName("image_url")
    val imageUrl: String? = null,
    @SerializedName("image_front_url")
    val imageFrontUrl: String? = null,
    @SerializedName("ingredients_text")
    val ingredientsText: String? = null,
    @SerializedName("nutriscore_grade")
    val nutriscoreGrade: String? = null,
    @SerializedName("nova_group")
    val novaGroup: Int? = null,
    @SerializedName("ecoscore_grade")
    val ecoscoreGrade: String? = null,
    @SerializedName("categories")
    val categories: String? = null,
    @SerializedName("allergens")
    val allergens: String? = null,
    @SerializedName("additives_tags")
    val additivesTags: List<String>? = null,
    @SerializedName("nutriments")
    val nutriments: ApiNutriments? = null
)
