package com.example.foodfacts.ui.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.navigation.fragment.findNavController
import coil.load
import coil.transform.RoundedCornersTransformation
import com.example.foodfacts.R
import com.example.foodfacts.data.model.ProductDomainModel
import com.example.foodfacts.databinding.FragmentProductDetailBinding
import com.example.foodfacts.ui.viewmodels.AiPredictionState
import com.example.foodfacts.ui.viewmodels.ProductDetailUiState
import com.example.foodfacts.ui.viewmodels.ProductDetailViewModel

class ProductDetailFragment : Fragment() {

    private var _binding: FragmentProductDetailBinding? = null
    private val binding get() = _binding!!

    private val viewModel: ProductDetailViewModel by viewModels()
    private var currentProduct: ProductDomainModel? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentProductDetailBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val barcode = arguments?.getString("barcode").orEmpty()
        if (barcode.isNotEmpty()) {
            viewModel.loadProduct(barcode)
        } else {
            Toast.makeText(requireContext(), "No product barcode provided", Toast.LENGTH_SHORT).show()
        }

        binding.btnCompareThis.setOnClickListener {
            currentProduct?.let { product ->
                val bundle = Bundle().apply {
                    putString("productABarcode", product.barcode)
                }
                findNavController().navigate(R.id.action_productDetailFragment_to_compareFragment, bundle)
            }
        }

        binding.btnPredictNovaAi.setOnClickListener {
            currentProduct?.let { product ->
                viewModel.predictNovaWithAi(product.ingredients)
            }
        }

        observeViewModel()
    }

    private fun observeViewModel() {
        viewModel.uiState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is ProductDetailUiState.Loading -> {
                    binding.progressBar.visibility = View.VISIBLE
                }
                is ProductDetailUiState.Success -> {
                    binding.progressBar.visibility = View.GONE
                    displayProduct(state.product)
                }
                is ProductDetailUiState.Error -> {
                    binding.progressBar.visibility = View.GONE
                    Toast.makeText(requireContext(), state.message, Toast.LENGTH_LONG).show()
                    if (state.cachedProduct != null) {
                        displayProduct(state.cachedProduct)
                    }
                }
            }
        }

        viewModel.aiPredictionState.observe(viewLifecycleOwner) { aiState ->
            when (aiState) {
                is AiPredictionState.Idle -> {
                    binding.tvAiPredictionResult.visibility = View.GONE
                }
                is AiPredictionState.Loading -> {
                    binding.tvAiPredictionResult.visibility = View.VISIBLE
                    binding.tvAiPredictionResult.text = "Sending ingredient text to Member 2 AI model..."
                }
                is AiPredictionState.Success -> {
                    binding.tvAiPredictionResult.visibility = View.VISIBLE
                    val pred = aiState.prediction
                    val groupStr = pred.predictedNovaGroup?.let { "NOVA Group $it" } ?: "NOVA Unknown"
                    val confStr = pred.confidenceScore?.let { " (Confidence: ${"%.1f".format(it * 100)}%)" } ?: ""
                    binding.tvAiPredictionResult.text = "[AI Model Prediction - ${pred.modelVersion}]\n$groupStr$confStr\nExplanation: ${pred.explanation ?: "Analyzed based on ingredients text."}"
                }
                is AiPredictionState.Error -> {
                    binding.tvAiPredictionResult.visibility = View.VISIBLE
                    binding.tvAiPredictionResult.text = "AI Service Status:\n${aiState.message}"
                }
            }
        }
    }

    private fun displayProduct(product: ProductDomainModel) {
        currentProduct = product

        binding.tvDetailTitle.text = product.name
        binding.tvDetailBrand.text = "${product.brand.ifEmpty { "Unknown Brand" }} • Barcode: ${product.barcode}"
        binding.tvDataSource.text = "Data Source: ${product.dataSource}"

        binding.imgProductLarge.load(product.imageUrl) {
            crossfade(true)
            placeholder(R.drawable.ic_launcher_foreground)
            error(R.drawable.ic_launcher_foreground)
            transformations(RoundedCornersTransformation(16f))
        }

        // Nutri-Score
        val nutri = product.nutriScore ?: "-"
        binding.badgeNutriScore.text = "Nutri-Score $nutri"
        val nutriColor = when (nutri) {
            "A" -> ContextCompat.getColor(requireContext(), android.R.color.holo_green_dark)
            "B" -> ContextCompat.getColor(requireContext(), android.R.color.holo_green_light)
            "C" -> ContextCompat.getColor(requireContext(), android.R.color.holo_orange_light)
            "D" -> ContextCompat.getColor(requireContext(), android.R.color.holo_orange_dark)
            "E" -> ContextCompat.getColor(requireContext(), android.R.color.holo_red_dark)
            else -> ContextCompat.getColor(requireContext(), android.R.color.darker_gray)
        }
        binding.badgeNutriScore.setBackgroundColor(nutriColor)

        // NOVA Group
        val nova = product.novaGroup
        binding.badgeNovaGroup.text = if (nova != null) "NOVA Group $nova" else "NOVA -"

        val novaDesc = when (nova) {
            1 -> "NOVA 1: Unprocessed or Minimally Processed Foods (Fresh, dried, frozen foods)."
            2 -> "NOVA 2: Processed Culinary Ingredients (Oils, butter, sugar, salt)."
            3 -> "NOVA 3: Processed Foods (Canned foods, simple cheeses, freshly baked breads)."
            4 -> "NOVA 4: Ultra-Processed Foods (Industrial formulations with additives & emulsifiers)."
            else -> "NOVA Classification: Not provided in Open Food Facts database."
        }
        binding.tvNovaClassificationDesc.text = "$novaDesc\n\nNote: Nutri-Score (A-E) measures nutritional quality, whereas NOVA (1-4) measures industrial food processing."

        // Eco-Score
        val eco = product.ecoScore ?: "-"
        binding.badgeEcoScore.text = "Eco-Score $eco"

        // Nutrition Facts
        binding.tvEnergy.text = "Energy: ${product.energyKcal?.let { "%.1f kcal".format(it) } ?: "N/A"}"
        binding.tvFat.text = "Fat: ${product.fat?.let { "%.1f g".format(it) } ?: "N/A"} (Saturated: ${product.saturatedFat?.let { "%.1f g".format(it) } ?: "N/A"})"
        binding.tvCarbs.text = "Carbohydrates: ${product.carbohydrates?.let { "%.1f g".format(it) } ?: "N/A"} (Sugars: ${product.sugars?.let { "%.1f g".format(it) } ?: "N/A"})"
        binding.tvFiber.text = "Dietary Fiber: ${product.fiber?.let { "%.1f g".format(it) } ?: "N/A"}"
        binding.tvProteins.text = "Proteins: ${product.proteins?.let { "%.1f g".format(it) } ?: "N/A"}"
        binding.tvSalt.text = "Salt: ${product.salt?.let { "%.3f g".format(it) } ?: "N/A"}"

        // Ingredients & Allergens & Additives
        binding.tvIngredientsText.text = product.ingredients.orEmpty().ifEmpty { "No ingredient information provided." }
        
        val allergens = product.allergens.orEmpty()
        if (allergens.isNotEmpty()) {
            binding.tvAllergensText.visibility = View.VISIBLE
            binding.tvAllergensText.text = "Allergens: $allergens"
        } else {
            binding.tvAllergensText.visibility = View.GONE
        }

        val additives = product.additives.orEmpty()
        if (additives.isNotEmpty()) {
            binding.tvAdditivesText.visibility = View.VISIBLE
            binding.tvAdditivesText.text = "Additives: $additives"
        } else {
            binding.tvAdditivesText.visibility = View.GONE
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
