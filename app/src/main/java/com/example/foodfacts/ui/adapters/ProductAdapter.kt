package com.example.foodfacts.ui.adapters

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import coil.load
import coil.transform.RoundedCornersTransformation
import com.example.foodfacts.R
import com.example.foodfacts.data.model.ProductDomainModel
import com.example.foodfacts.databinding.ItemProductBinding

class ProductAdapter(
    private val onItemClick: (ProductDomainModel) -> Unit,
    private val onItemLongClick: ((ProductDomainModel) -> Unit)? = null
) : ListAdapter<ProductDomainModel, ProductAdapter.ProductViewHolder>(DiffCallback) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ProductViewHolder {
        val binding = ItemProductBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return ProductViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ProductViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ProductViewHolder(
        private val binding: ItemProductBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(product: ProductDomainModel) {
            binding.tvProductName.text = product.name
            binding.tvBrand.text = product.brand.ifEmpty { "Unknown Brand" }
            binding.tvBarcode.text = "Barcode: ${product.barcode}"

            binding.imgProduct.load(product.imageUrl) {
                crossfade(true)
                placeholder(R.drawable.ic_launcher_foreground)
                error(R.drawable.ic_launcher_foreground)
                transformations(RoundedCornersTransformation(12f))
            }

            // Nutri-Score Badge
            val nutri = product.nutriScore ?: "-"
            binding.tvNutriScoreBadge.text = "Nutri-Score $nutri"
            val nutriColor = when (nutri.uppercase()) {
                "A" -> ContextCompat.getColor(binding.root.context, android.R.color.holo_green_dark)
                "B" -> ContextCompat.getColor(binding.root.context, android.R.color.holo_green_light)
                "C" -> ContextCompat.getColor(binding.root.context, android.R.color.holo_orange_light)
                "D" -> ContextCompat.getColor(binding.root.context, android.R.color.holo_orange_dark)
                "E" -> ContextCompat.getColor(binding.root.context, android.R.color.holo_red_dark)
                else -> ContextCompat.getColor(binding.root.context, android.R.color.darker_gray)
            }
            binding.tvNutriScoreBadge.setBackgroundColor(nutriColor)

            // NOVA Group Badge
            binding.tvNovaBadge.text = if (product.novaGroup != null) "NOVA ${product.novaGroup}" else "NOVA -"

            binding.root.setOnClickListener { onItemClick(product) }
            binding.root.setOnLongClickListener {
                onItemLongClick?.invoke(product)
                onItemLongClick != null
            }
        }
    }

    object DiffCallback : DiffUtil.ItemCallback<ProductDomainModel>() {
        override fun areItemsTheSame(oldItem: ProductDomainModel, newItem: ProductDomainModel): Boolean {
            return oldItem.barcode == newItem.barcode
        }

        override fun areContentsTheSame(oldItem: ProductDomainModel, newItem: ProductDomainModel): Boolean {
            return oldItem == newItem
        }
    }
}
