package com.example.foodfacts.ui.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import coil.load
import com.example.foodfacts.R
import com.google.android.material.card.MaterialCardView

class HomeFragment : Fragment() {

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_home, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val cardScan = view.findViewById<MaterialCardView>(R.id.cardScan)
        val cardSearch = view.findViewById<MaterialCardView>(R.id.cardSearch)
        val cardOcr = view.findViewById<MaterialCardView>(R.id.cardOcr)
        val cardCompare = view.findViewById<MaterialCardView>(R.id.cardCompare)
        val cardHistory = view.findViewById<MaterialCardView>(R.id.cardHistory)

        val ivScanBg = view.findViewById<ImageView>(R.id.ivScanBg)
        val ivSearchBg = view.findViewById<ImageView>(R.id.ivSearchBg)
        val ivOcrBg = view.findViewById<ImageView>(R.id.ivOcrBg)
        val ivCompareBg = view.findViewById<ImageView>(R.id.ivCompareBg)
        val ivHistoryBg = view.findViewById<ImageView>(R.id.ivHistoryBg)

        // Load beautiful Unsplash images for "Wow" moment
        ivScanBg.load("https://images.unsplash.com/photo-1556740738-b6a63e27c4df?q=80&w=500&auto=format&fit=crop") {
            crossfade(true)
        }
        ivSearchBg.load("https://images.unsplash.com/photo-1542838132-92c53300491e?q=80&w=500&auto=format&fit=crop") {
            crossfade(true)
        }
        ivOcrBg.load("https://images.unsplash.com/photo-1506617420156-8e4536971650?q=80&w=500&auto=format&fit=crop") {
            crossfade(true)
        }
        ivCompareBg.load("https://images.unsplash.com/photo-1607305387299-a3d9611cd469?q=80&w=500&auto=format&fit=crop") {
            crossfade(true)
        }
        ivHistoryBg.load("https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=500&auto=format&fit=crop") {
            crossfade(true)
        }

        // Navigation
        cardScan.setOnClickListener {
            findNavController().navigate(R.id.action_homeFragment_to_scanFragment)
        }

        cardSearch.setOnClickListener {
            findNavController().navigate(R.id.action_homeFragment_to_searchFragment)
        }

        cardOcr.setOnClickListener {
            findNavController().navigate(R.id.action_homeFragment_to_ocrFragment)
        }

        cardCompare.setOnClickListener {
            findNavController().navigate(R.id.action_homeFragment_to_compareFragment)
        }

        cardHistory.setOnClickListener {
            findNavController().navigate(R.id.action_homeFragment_to_historyFragment)
        }
    }
}
