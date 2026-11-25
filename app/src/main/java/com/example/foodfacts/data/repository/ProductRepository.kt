package com.example.foodfacts.data.repository

import com.example.foodfacts.data.api.NetworkClient
import com.example.foodfacts.data.api.OpenFoodFactsApi
import com.example.foodfacts.data.api.model.ApiProduct
import com.example.foodfacts.data.local.ProductDao
import com.example.foodfacts.data.local.ScanEventDao
import com.example.foodfacts.data.local.toDomainModel
import com.example.foodfacts.data.local.toEntity
import com.example.foodfacts.data.model.ComparisonResult
import com.example.foodfacts.data.model.ComparisonWinner
import com.example.foodfacts.data.model.MetricComparison
import com.example.foodfacts.data.model.ProductDomainModel
import com.example.foodfacts.data.model.Resource
import com.example.foodfacts.data.model.ScanEvent
import com.example.foodfacts.data.model.ScanSource
import com.example.foodfacts.integration.events.ScanEventLogger
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.util.Locale

class ProductRepository(
    private val productDao: ProductDao,
    private val scanEventDao: ScanEventDao,
    private val api: OpenFoodFactsApi = NetworkClient.api
) {
    private val eventLogger = ScanEventLogger(scanEventDao)

    suspend fun getProductByBarcode(
        barcode: String,
        scanSource: ScanSource = ScanSource.BARCODE
    ): Resource<ProductDomainModel> {
        val cleanBarcode = barcode.trim()
        if (cleanBarcode.isEmpty()) {
            return Resource.Error("Invalid barcode string")
        }

        val cachedEntity = try {
            productDao.getProductByBarcode(cleanBarcode)
        } catch (_: Exception) {
            null
        }

        return try {
            val response = api.getProductByBarcode(cleanBarcode)
            if (response.isSuccessful) {
                val body = response.body()
                val apiProduct = body?.product
                if (body?.status == 1 && apiProduct != null) {
                    val domainModel = mapApiProductToDomain(cleanBarcode, apiProduct)
                    productDao.insertOrUpdate(domainModel.toEntity())
                    eventLogger.logProductLookupEvent(
                        barcode = cleanBarcode,
                        scanSource = scanSource,
                        product = domainModel,
                        isNetworkError = false
                    )
                    Resource.Success(domainModel)
                } else {
                    eventLogger.logProductLookupEvent(
                        barcode = cleanBarcode,
                        scanSource = scanSource,
                        product = null,
                        isNetworkError = false
                    )
                    if (cachedEntity != null) {
                        Resource.Success(cachedEntity.toDomainModel())
                    } else {
                        Resource.Error("Product not found on Open Food Facts (Barcode: $cleanBarcode)")
                    }
                }
            } else {
                if (cachedEntity != null) {
                    Resource.Success(cachedEntity.toDomainModel())
                } else {
                    Resource.Error("Server error (${response.code()}): ${response.message()}")
                }
            }
        } catch (e: Exception) {
            eventLogger.logProductLookupEvent(
                barcode = cleanBarcode,
                scanSource = scanSource,
                product = cachedEntity?.toDomainModel(),
                isNetworkError = true
            )
            if (cachedEntity != null) {
                Resource.Error(
                    message = "Network unavailable. Showing cached product details.",
                    cause = e,
                    cachedData = cachedEntity.toDomainModel()
                )
            } else {
                Resource.Error(
                    message = "Network request failed. Please check connection.",
                    cause = e
                )
            }
        }
    }

    suspend fun searchProducts(query: String): Resource<List<ProductDomainModel>> {
        val cleanQuery = query.trim()
        if (cleanQuery.isEmpty()) {
            return Resource.Success(emptyList())
        }
        return try {
            val response = api.searchProducts(searchTerms = cleanQuery)
            if (response.isSuccessful) {
                val apiProducts = response.body()?.products ?: emptyList()
                val domainProducts = apiProducts.map { apiProduct ->
                    val code = apiProduct.code ?: System.currentTimeMillis().toString()
                    mapApiProductToDomain(code, apiProduct)
                }
                Resource.Success(domainProducts)
            } else {
                Resource.Error("Search failed with server code ${response.code()}")
            }
        } catch (e: Exception) {
            Resource.Error("Search error: ${e.localizedMessage ?: "Connection error"}", cause = e)
        }
    }

    fun getHistoryProductsFlow(): Flow<List<ProductDomainModel>> {
        return productDao.getAllProductsFlow().map { entities ->
            entities.map { it.toDomainModel() }
        }
    }

    fun getScanEventsFlow(): Flow<List<ScanEvent>> {
        return scanEventDao.getAllEventsFlow().map { entities ->
            entities.map { it.toDomainModel() }
        }
    }

    suspend fun recordScanEvent(
        barcode: String,
        productName: String,
        brand: String,
        scanSource: ScanSource
    ) {
        val cleanBarcode = barcode.trim()
        val event = ScanEvent(
            barcode = cleanBarcode,
            normalizedBarcode = cleanBarcode.padStart(14, '0'),
            productName = productName.ifEmpty { "Unknown Product" },
            brand = brand.ifEmpty { "Unknown Brand" },
            productFound = true,
            novaSource = "OPEN_FOOD_FACTS",
            scanSource = scanSource,
            eventSource = "ANDROID_APP",
            lookupStatus = "MANUAL_RECORD",
            timestamp = System.currentTimeMillis()
        )
        eventLogger.logCustomEvent(event)
    }

    suspend fun deleteHistoryProduct(barcode: String) {
        productDao.deleteByBarcode(barcode)
    }

    suspend fun clearHistory() {
        productDao.clearAll()
        scanEventDao.clearAll()
    }

    fun compareProducts(
        productA: ProductDomainModel,
        productB: ProductDomainModel
    ): ComparisonResult {
        val metrics = mutableListOf<MetricComparison>()

        // 1. Nutri-Score Comparison
        val nutriWinner = compareGrades(productA.nutriScore, productB.nutriScore)
        metrics.add(
            MetricComparison(
                metricName = "Nutri-Score",
                valueA = gradeToNumeric(productA.nutriScore),
                valueB = gradeToNumeric(productB.nutriScore),
                unit = "Grade (A is best)",
                lowerIsBetter = true,
                winner = nutriWinner
            )
        )

        // 2. NOVA Group Comparison
        val novaWinner = compareNumericLowerBetter(productA.novaGroup?.toDouble(), productB.novaGroup?.toDouble())
        metrics.add(
            MetricComparison(
                metricName = "NOVA Group",
                valueA = productA.novaGroup?.toDouble(),
                valueB = productB.novaGroup?.toDouble(),
                unit = "Group (1: Unprocessed - 4: Ultra-processed)",
                lowerIsBetter = true,
                winner = novaWinner
            )
        )

        // 3. Energy Kcal
        val energyWinner = compareNumericLowerBetter(productA.energyKcal, productB.energyKcal)
        metrics.add(
            MetricComparison(
                metricName = "Energy",
                valueA = productA.energyKcal,
                valueB = productB.energyKcal,
                unit = "kcal / 100g",
                lowerIsBetter = true,
                winner = energyWinner
            )
        )

        // 4. Sugars
        val sugarWinner = compareNumericLowerBetter(productA.sugars, productB.sugars)
        metrics.add(
            MetricComparison(
                metricName = "Sugars",
                valueA = productA.sugars,
                valueB = productB.sugars,
                unit = "g / 100g",
                lowerIsBetter = true,
                winner = sugarWinner
            )
        )

        // 5. Saturated Fat
        val satFatWinner = compareNumericLowerBetter(productA.saturatedFat, productB.saturatedFat)
        metrics.add(
            MetricComparison(
                metricName = "Saturated Fat",
                valueA = productA.saturatedFat,
                valueB = productB.saturatedFat,
                unit = "g / 100g",
                lowerIsBetter = true,
                winner = satFatWinner
            )
        )

        // 6. Salt
        val saltWinner = compareNumericLowerBetter(productA.salt, productB.salt)
        metrics.add(
            MetricComparison(
                metricName = "Salt",
                valueA = productA.salt,
                valueB = productB.salt,
                unit = "g / 100g",
                lowerIsBetter = true,
                winner = saltWinner
            )
        )

        // 7. Fiber (Higher is better)
        val fiberWinner = compareNumericHigherBetter(productA.fiber, productB.fiber)
        metrics.add(
            MetricComparison(
                metricName = "Dietary Fiber",
                valueA = productA.fiber,
                valueB = productB.fiber,
                unit = "g / 100g",
                lowerIsBetter = false,
                winner = fiberWinner
            )
        )

        // 8. Proteins (Higher is better)
        val proteinWinner = compareNumericHigherBetter(productA.proteins, productB.proteins)
        metrics.add(
            MetricComparison(
                metricName = "Proteins",
                valueA = productA.proteins,
                valueB = productB.proteins,
                unit = "g / 100g",
                lowerIsBetter = false,
                winner = proteinWinner
            )
        )

        // Overall Winner Score Calculation
        var scoreA = 0
        var scoreB = 0
        metrics.forEach { metric ->
            when (metric.winner) {
                ComparisonWinner.PRODUCT_A -> scoreA++
                ComparisonWinner.PRODUCT_B -> scoreB++
                else -> {}
            }
        }

        val overallWinner = when {
            scoreA > scoreB -> ComparisonWinner.PRODUCT_A
            scoreB > scoreA -> ComparisonWinner.PRODUCT_B
            else -> ComparisonWinner.TIE
        }

        val summary = when (overallWinner) {
            ComparisonWinner.PRODUCT_A -> "${productA.name} is recommended overall ($scoreA better metrics vs $scoreB)."
            ComparisonWinner.PRODUCT_B -> "${productB.name} is recommended overall ($scoreB better metrics vs $scoreA)."
            else -> "Both products have comparable overall nutritional performance."
        }

        return ComparisonResult(
            productA = productA,
            productB = productB,
            metrics = metrics,
            recommendedProduct = overallWinner,
            summary = summary
        )
    }

    private fun mapApiProductToDomain(code: String, apiProduct: ApiProduct): ProductDomainModel {
        val nutriments = apiProduct.nutriments
        val kcal = nutriments?.energyKcal100g ?: nutriments?.energyKcal
        val additivesStr = apiProduct.additivesTags?.takeIf { it.isNotEmpty() }?.joinToString(", ") { tag ->
            tag.removePrefix("en:").replace('-', ' ').uppercase(Locale.ROOT)
        }

        return ProductDomainModel(
            barcode = code,
            name = apiProduct.productName.orEmpty().ifEmpty { "Unnamed Product" },
            brand = apiProduct.brands.orEmpty().ifEmpty { "Unknown Brand" },
            imageUrl = apiProduct.imageFrontUrl ?: apiProduct.imageUrl,
            nutriScore = apiProduct.nutriscoreGrade?.uppercase(Locale.ROOT),
            novaGroup = apiProduct.novaGroup,
            ecoScore = apiProduct.ecoscoreGrade?.uppercase(Locale.ROOT),
            ingredients = apiProduct.ingredientsText,
            categories = apiProduct.categories,
            allergens = apiProduct.allergens,
            additives = additivesStr,
            energyKcal = kcal,
            fat = nutriments?.fat100g,
            saturatedFat = nutriments?.saturatedFat100g,
            carbohydrates = nutriments?.carbohydrates100g,
            sugars = nutriments?.sugars100g,
            fiber = nutriments?.fiber100g,
            proteins = nutriments?.proteins100g,
            salt = nutriments?.salt100g,
            dataSource = "Open Food Facts Database",
            lastAccessedAt = System.currentTimeMillis()
        )
    }

    private fun compareGrades(gradeA: String?, gradeB: String?): ComparisonWinner {
        val scoreA = gradeToNumeric(gradeA)
        val scoreB = gradeToNumeric(gradeB)
        if (scoreA == null && scoreB == null) return ComparisonWinner.TIE
        if (scoreA == null) return ComparisonWinner.PRODUCT_B
        if (scoreB == null) return ComparisonWinner.PRODUCT_A
        return when {
            scoreA < scoreB -> ComparisonWinner.PRODUCT_A
            scoreB < scoreA -> ComparisonWinner.PRODUCT_B
            else -> ComparisonWinner.TIE
        }
    }

    private fun gradeToNumeric(grade: String?): Double? {
        return when (grade?.uppercase(Locale.ROOT)) {
            "A" -> 1.0
            "B" -> 2.0
            "C" -> 3.0
            "D" -> 4.0
            "E" -> 5.0
            else -> null
        }
    }

    private fun compareNumericLowerBetter(valA: Double?, valB: Double?): ComparisonWinner {
        if (valA == null && valB == null) return ComparisonWinner.TIE
        if (valA == null) return ComparisonWinner.PRODUCT_B
        if (valB == null) return ComparisonWinner.PRODUCT_A
        return when {
            valA < valB -> ComparisonWinner.PRODUCT_A
            valB < valA -> ComparisonWinner.PRODUCT_B
            else -> ComparisonWinner.TIE
        }
    }

    private fun compareNumericHigherBetter(valA: Double?, valB: Double?): ComparisonWinner {
        if (valA == null && valB == null) return ComparisonWinner.TIE
        if (valA == null) return ComparisonWinner.PRODUCT_B
        if (valB == null) return ComparisonWinner.PRODUCT_A
        return when {
            valA > valB -> ComparisonWinner.PRODUCT_A
            valB > valA -> ComparisonWinner.PRODUCT_B
            else -> ComparisonWinner.TIE
        }
    }
}
