# Visualization Best Practices and Common Issues

## Documentation of Fixes and Guidelines for Future Plot Creation

### ✅ **DO's - Best Practices**

#### **Plot Layout and Spacing**
- **Use adequate subplot spacing**: `plt.subplots_adjust(hspace=0.4, wspace=0.3)` or `plt.tight_layout(pad=2.0)`
- **Set proper figure sizes**: Minimum 15x12 for multi-panel plots, 10x8 for single plots
- **Add padding around scatter plots**: Use 15-20% padding beyond data range to prevent edge cutoff
- **Position legends carefully**: Use `bbox_to_anchor` for precise legend positioning outside plot area

#### **Title and Text Management**
- **Avoid title overlap**: Set main title `y=0.98` and subplot titles with adequate spacing
- **Use consistent font sizes**: Main title 16-18pt, subplot titles 14pt, labels 12pt
- **Keep text outside plots**: Place detailed explanations in markdown, not on plots
- **Test title positioning**: Always check for overlaps before finalizing

#### **Color and Legend Consistency**
- **Match legend to actual data**: Ensure all plotted elements appear in legend
- **Use consistent color schemes**: Blue for Gaussian, Red for alternatives, Black for theoretical
- **Limit color palette**: Maximum 4-5 colors per plot for clarity
- **Add transparency**: Use alpha=0.7 for overlapping data points

#### **Data Point Management**
- **Reduce sample size for clarity**: Use 800-2000 points for scatter plots instead of 5000+
- **Filter extreme outliers**: Use percentile-based limits (5th-95th percentile) for display
- **Maintain statistical accuracy**: Keep full dataset for calculations, subsample for visualization only

#### **3D Plot Guidelines**
- **Use sparingly**: One representative 3D plot per document section
- **Focus on key insights**: Choose the most informative distribution for 3D representation
- **Ensure readability**: Test viewing angles and color schemes for clarity

### ❌ **DON'Ts - Common Issues to Avoid**

#### **Layout Problems**
- **Don't overlap titles**: Main title too close to subplot titles
- **Don't crowd subplots**: Insufficient spacing between panels
- **Don't cut off data**: Inadequate margins around scatter plot boundaries
- **Don't use default spacing**: Always customize `hspace`, `wspace`, and `pad` parameters

#### **Legend and Color Issues**
- **Don't mismatch legends**: All plotted elements must be represented
- **Don't use unclear colors**: Avoid similar colors that are hard to distinguish
- **Don't hide legends**: Ensure legends don't cover important plot elements
- **Don't ignore transparency**: Overlapping elements need alpha adjustment

#### **Text and Annotation Problems**
- **Don't clutter plots with text**: Move explanations to markdown documentation
- **Don't use inconsistent fonts**: Maintain font hierarchy throughout
- **Don't place text over data**: Ensure annotations don't obscure important information
- **Don't forget accessibility**: Consider colorblind-friendly palettes

#### **Data Visualization Errors**
- **Don't oversample**: Too many points create visual noise
- **Don't undersample**: Too few points miss distribution characteristics
- **Don't ignore outliers completely**: Show some outliers but prevent scale distortion
- **Don't use inappropriate scales**: Match scale to data characteristics

### 🔧 **Specific Fixes Applied**

#### **Issue 1: Text Covering Legend**
- **Problem**: Annotation text overlapping legend in contour plots
- **Solution**: Remove annotations or position them using `bbox_to_anchor` away from legend area
- **Code**: `ax.annotate(..., xy=safe_position, bbox=dict(boxstyle="round,pad=0.5", alpha=0.8))`

#### **Issue 2: Title Overlap**
- **Problem**: Main title overlapping subplot titles
- **Solution**: Increase spacing with `fig.suptitle(..., y=0.98)` and `plt.subplots_adjust(top=0.92)`
- **Code**: `fig.suptitle('Title', y=0.98)` + `plt.tight_layout(pad=2.0)`

#### **Issue 3: Legend-Color Mismatch**
- **Problem**: Three colors plotted but only two in legend
- **Solution**: Ensure every plot element has corresponding legend entry
- **Code**: Check all `ax.plot()`, `ax.scatter()`, `ax.hist()` calls have `label` parameter

#### **Issue 4: Scatter Plot Margins**
- **Problem**: Elliptical contours appearing cropped at edges
- **Solution**: Add 20% padding beyond data range + wider axis limits
- **Code**: `padding = (data_range) * 0.2; ax.set_xlim(min_val - padding, max_val + padding)`

#### **Issue 5: Insufficient Generator Function Plots**
- **Problem**: Only one generator function comparison plot
- **Solution**: Create multiple plots showing different aspects of generator functions
- **Implementation**: Separate plots for basic comparison, log scale, tail behavior, and mathematical properties

#### **Issue 6: Excessive 3D Plots**
- **Problem**: Too many 3D visualizations causing clutter
- **Solution**: Reduce to one representative 3D plot per major section
- **Guideline**: Choose most informative case (e.g., Student's t ν=3 for heavy tail demonstration)

### 📋 **Checklist for New Plots**

Before finalizing any visualization:

- [ ] Check title positioning (no overlaps)
- [ ] Verify legend matches all plotted elements  
- [ ] Ensure adequate subplot spacing
- [ ] Test scatter plot margins (no edge cutoff)
- [ ] Validate color scheme consistency
- [ ] Confirm text readability and positioning
- [ ] Review data point density (not too crowded/sparse)
- [ ] Test different screen sizes/resolutions
- [ ] Verify accessibility (colorblind-friendly)
- [ ] Document any custom parameters used

### 🎯 **Future Improvement Guidelines**

1. **Always test layouts** on different screen sizes before committing
2. **Use parameterized spacing** to easily adjust layouts across similar plots
3. **Create plot templates** for consistent styling across projects
4. **Document custom parameters** in code comments for future reference
5. **Regular review** of visualization guidelines as standards evolve

This document should be updated whenever new visualization issues are discovered or resolved.