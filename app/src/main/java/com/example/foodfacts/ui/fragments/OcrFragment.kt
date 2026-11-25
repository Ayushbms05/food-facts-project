package com.example.foodfacts.ui.fragments

import android.graphics.Bitmap
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.navigation.fragment.findNavController
import com.example.foodfacts.R
import com.example.foodfacts.databinding.FragmentOcrBinding
import com.example.foodfacts.ui.viewmodels.AiPredictionState
import com.example.foodfacts.ui.viewmodels.OcrUiState
import com.example.foodfacts.ui.viewmodels.OcrViewModel

class OcrFragment : Fragment() {

    private var _binding: FragmentOcrBinding? = null
    private val binding get() = _binding!!

    private val viewModel: OcrViewModel by viewModels()

    private val getContentLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            binding.imgOcrPreview.visibility = View.VISIBLE
            binding.imgOcrPreview.setImageURI(uri)
            viewModel.processImageUri(uri)
        }
    }

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap: Bitmap? ->
        if (bitmap != null) {
            binding.imgOcrPreview.visibility = View.VISIBLE
            binding.imgOcrPreview.setImageBitmap(bitmap)
            viewModel.processBitmap(bitmap)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentOcrBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupListeners()
        observeViewModel()
    }

    private fun setupListeners() {
        binding.btnPickGallery.setOnClickListener {
            getContentLauncher.launch("image/*")
        }

        binding.btnCaptureCamera.setOnClickListener {
            takePictureLauncher.launch(null)
        }

        binding.btnRetryOcr.setOnClickListener {
            binding.imgOcrPreview.visibility = View.GONE
            binding.etExtractedIngredients.setText("")
            binding.cardAiPredictionResult.visibility = View.GONE
            binding.cardAllergens.visibility = View.GONE
            viewModel.reset()
        }

        binding.btnPredictNovaAi.setOnClickListener {
            val text = binding.etExtractedIngredients.text?.toString().orEmpty()
            viewModel.predictNovaWithAi(text)
        }

        binding.btnSearchExtractedText.setOnClickListener {
            val text = binding.etExtractedIngredients.text?.toString().orEmpty()
            if (text.isNotEmpty()) {
                val searchWord = text.split("\\s+".toRegex()).firstOrNull { it.length > 3 } ?: text
                val bundle = Bundle().apply {
                    putString("ocrQuery", searchWord)
                }
                findNavController().navigate(R.id.action_ocrFragment_to_searchFragment, bundle)
            } else {
                Toast.makeText(requireContext(), "Ingredients text is empty", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun observeViewModel() {
        viewModel.uiState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is OcrUiState.Idle -> {
                    binding.progressBar.visibility = View.GONE
                    binding.cardAllergens.visibility = View.GONE
                }
                is OcrUiState.Processing -> {
                    binding.progressBar.visibility = View.VISIBLE
                }
                is OcrUiState.Success -> {
                    binding.progressBar.visibility = View.GONE
                    binding.etExtractedIngredients.setText(state.extractedText)

                    if (state.detectedAllergens.isNotEmpty()) {
                        binding.cardAllergens.visibility = View.VISIBLE
                        binding.tvDetectedAllergens.text = state.detectedAllergens.joinToString(", ")
                    } else {
                        binding.cardAllergens.visibility = View.GONE
                    }
                }
                is OcrUiState.Error -> {
                    binding.progressBar.visibility = View.GONE
                    Toast.makeText(requireContext(), state.message, Toast.LENGTH_SHORT).show()
                }
            }
        }

        viewModel.aiPredictionState.observe(viewLifecycleOwner) { aiState ->
            when (aiState) {
                is AiPredictionState.Idle -> {
                    binding.cardAiPredictionResult.visibility = View.GONE
                }
                is AiPredictionState.Loading -> {
                    binding.cardAiPredictionResult.visibility = View.VISIBLE
                    binding.tvAiPredictionResult.text = "Predicting NOVA Group with Member 2 AI model..."
                }
                is AiPredictionState.Success -> {
                    binding.cardAiPredictionResult.visibility = View.VISIBLE
                    val pred = aiState.prediction
                    val groupStr = pred.predictedNovaGroup?.let { "NOVA Group $it" } ?: "NOVA Group Unknown"
                    val confStr = pred.confidenceScore?.let { " (Confidence: ${"%.1f".format(it * 100)}%)" } ?: ""
                    binding.tvAiPredictionResult.text = "[AI Model Prediction - ${pred.modelVersion}]\n$groupStr$confStr\nExplanation: ${pred.explanation ?: "Analyzed based on ingredients text."}"
                }
                is AiPredictionState.Error -> {
                    binding.cardAiPredictionResult.visibility = View.VISIBLE
                    binding.tvAiPredictionResult.text = "AI Service Status:\n${aiState.message}"
                }
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
