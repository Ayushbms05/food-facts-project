package com.example.foodfacts.data.api.model

import com.google.gson.annotations.SerializedName

data class SearchResponse(
    @SerializedName("count")
    val count: Int? = null,
    @SerializedName("page")
    val page: Int? = null,
    @SerializedName("page_size")
    val pageSize: Int? = null,
    @SerializedName("products")
    val products: List<ApiProduct>? = null
)
