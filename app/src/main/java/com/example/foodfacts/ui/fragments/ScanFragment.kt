package com.example.foodfacts.ui.fragments

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.navigation.fragment.findNavController
import com.example.foodfacts.R
import com.example.foodfacts.databinding.FragmentScanBinding
import com.example.foodfacts.ui.viewmodels.ScanState
import com.example.foodfacts.ui.viewmodels.ScanViewModel
import com.example.foodfacts.util.BarcodeAnalyzer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class ScanFragment : Fragment() {

    private var _binding: FragmentScanBinding? = null
    private val binding get() = _binding!!

    private val viewModel: ScanViewModel by viewModels()
    private lateinit var cameraExecutor: ExecutorService

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(requireContext(), "Camera permission required for scanning", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentScanBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()

        checkCameraPermission()
        setupListeners()
        observeViewModel()
    }

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(binding.previewView.surfaceProvider)
                    }

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor, BarcodeAnalyzer { barcode ->
                            activity?.runOnUiThread {
                                viewModel.onBarcodeScanned(barcode)
                            }
                        })
                    }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    viewLifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                )
            } catch (e: Exception) {
                Toast.makeText(requireContext(), "Failed to start camera: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun setupListeners() {
        binding.btnManualLookup.setOnClickListener {
            val barcode = binding.etManualBarcode.text?.toString()?.trim() ?: ""
            if (barcode.isNotEmpty()) {
                viewModel.fetchProductByBarcode(barcode)
            } else {
                Toast.makeText(requireContext(), "Please enter a valid barcode number", Toast.LENGTH_SHORT).show()
            }
        }

        binding.btnOcrScan.setOnClickListener {
            findNavController().navigate(R.id.action_scanFragment_to_ocrFragment)
        }
    }

    private fun observeViewModel() {
        viewModel.scanState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is ScanState.Idle -> {
                    binding.progressBar.visibility = View.GONE
                }
                is ScanState.Loading -> {
                    binding.progressBar.visibility = View.VISIBLE
                }
                is ScanState.Success -> {
                    binding.progressBar.visibility = View.GONE
                    val bundle = Bundle().apply {
                        putString("barcode", state.product.barcode)
                    }
                    viewModel.resetState()
                    findNavController().navigate(R.id.action_scanFragment_to_productDetailFragment, bundle)
                }
                is ScanState.Error -> {
                    binding.progressBar.visibility = View.GONE
                    if (state.cachedData != null) {
                        Toast.makeText(requireContext(), state.message, Toast.LENGTH_LONG).show()
                        val bundle = Bundle().apply {
                            putString("barcode", state.cachedData.barcode)
                        }
                        viewModel.resetState()
                        findNavController().navigate(R.id.action_scanFragment_to_productDetailFragment, bundle)
                    } else {
                        Toast.makeText(requireContext(), state.message, Toast.LENGTH_SHORT).show()
                        viewModel.resetState()
                    }
                }
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        cameraExecutor.shutdown()
        _binding = null
    }
}
