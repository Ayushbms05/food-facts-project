package com.example.foodfacts.data.api.model

import com.google.gson.annotations.SerializedName

data class ProductResponse(
    @SerializedName("code")
    val code: String? = null,
    @SerializedName("status")
    val status: Int? = null,
    @SerializedName("status_verbose")
    val statusVerbose: String? = null,
    @SerializedName("product")
    val product: ApiProduct? = null
)
