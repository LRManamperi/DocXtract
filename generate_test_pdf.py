"""
Generate test PDFs with tables and charts for DocXtract testing
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def create_test_pdf_with_charts():
    """Create a test PDF with various chart types"""
    
    pdf_path = "test_charts.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Bar Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ['Q1', 'Q2', 'Q3', 'Q4']
        values = [45, 72, 58, 81]
        bars = ax.bar(categories, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax.set_title('Quarterly Sales Report', fontsize=16, fontweight='bold')
        ax.set_xlabel('Quarter', fontsize=12)
        ax.set_ylabel('Sales (in thousands)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Line Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        revenue = [30, 45, 38, 55, 62, 58]
        ax.plot(months, revenue, marker='o', linewidth=2, markersize=8, color='#3498db')
        ax.set_title('Monthly Revenue Trend', fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Revenue ($K)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (month, val) in enumerate(zip(months, revenue)):
            ax.text(i, val + 2, str(val), ha='center', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 3: Pie Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['Product A', 'Product B', 'Product C', 'Product D']
        sizes = [35, 25, 20, 20]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        explode = (0.1, 0, 0, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('Market Share Distribution', fontsize=16, fontweight='bold')
        ax.axis('equal')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 4: Scatter Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        np.random.seed(42)
        x = np.random.randn(50) * 10 + 50
        y = x * 1.5 + np.random.randn(50) * 10
        ax.scatter(x, y, alpha=0.6, s=100, color='#e74c3c')
        ax.set_title('Performance vs Experience', fontsize=16, fontweight='bold')
        ax.set_xlabel('Experience (years)', fontsize=12)
        ax.set_ylabel('Performance Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print(f"✅ Created test PDF with charts: {pdf_path}")
    return pdf_path


def create_test_pdf_with_tables():
    """Create a test PDF with various table types"""
    
    pdf_path = "test_tables.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Structured table with grid lines
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        # Table data
        data = [
            ['Product', 'Q1', 'Q2', 'Q3', 'Q4'],
            ['Widget A', '100', '150', '200', '180'],
            ['Widget B', '80', '90', '110', '120'],
            ['Widget C', '120', '130', '140', '150'],
            ['Widget D', '95', '105', '125', '135']
        ]
        
        table = ax.table(cellText=data, cellLoc='center', loc='center',
                        colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, 5):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        ax.text(0.5, 0.95, 'Sales Data by Quarter', 
                ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Unstructured table (space-separated, no borders)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Employee Performance (No Grid Lines)', 
                ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        
        # Create text-based table
        table_text = """
        Name            Department      Score   Rating
        
        John Smith      Sales           92      A
        Jane Doe        Marketing       88      B+
        Bob Johnson     Engineering     95      A
        Alice Brown     HR              85      B
        Charlie Davis   Finance         90      A-
        """
        
        ax.text(0.5, 0.5, table_text, 
                ha='center', va='center', fontsize=11,
                family='monospace', transform=ax.transAxes)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 3: Mixed structured table
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        data2 = [
            ['Region', 'Revenue', 'Growth'],
            ['North', '$2.5M', '+15%'],
            ['South', '$1.8M', '+8%'],
            ['East', '$3.2M', '+22%'],
            ['West', '$2.1M', '+12%']
        ]
        
        table2 = ax.table(cellText=data2, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.3, 0.3])
        table2.auto_set_font_size(False)
        table2.set_fontsize(11)
        table2.scale(1, 2.5)
        
        # Style
        for i in range(3):
            table2[(0, i)].set_facecolor('#2ecc71')
            table2[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.text(0.5, 0.95, 'Regional Performance', 
                ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print(f"✅ Created test PDF with tables: {pdf_path}")
    return pdf_path


def create_combined_test_pdf():
    """Create a comprehensive test PDF with both tables and charts"""
    
    pdf_path = "test_combined.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Title page
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.text(0.5, 0.6, 'DocXtract Test Document', 
                ha='center', va='center', fontsize=24, fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.4, 'Contains Tables and Charts for Testing', 
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Table
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        data = [
            ['Month', 'Sales', 'Expenses', 'Profit'],
            ['Jan', '$45K', '$30K', '$15K'],
            ['Feb', '$52K', '$32K', '$20K'],
            ['Mar', '$48K', '$28K', '$20K'],
            ['Apr', '$61K', '$35K', '$26K']
        ]
        
        table = ax.table(cellText=data, cellLoc='center', loc='center',
                        colWidths=[0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(4):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.text(0.5, 0.95, 'Financial Summary', 
                ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 3: Bar Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        months = ['Jan', 'Feb', 'Mar', 'Apr']
        profits = [15, 20, 20, 26]
        ax.bar(months, profits, color='#2ecc71')
        ax.set_title('Monthly Profit', fontsize=16, fontweight='bold')
        ax.set_ylabel('Profit ($K)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(profits):
            ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 4: Line Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        weeks = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
        customers = [120, 145, 138, 162, 175, 168]
        ax.plot(weeks, customers, marker='o', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_title('Weekly Customer Count', fontsize=16, fontweight='bold')
        ax.set_ylabel('Customers', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 5: Another table (unstructured style)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Product Inventory', 
                ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        
        inventory_text = """
        ID      Product         Stock   Price
        
        101     Laptop          25      $899
        102     Mouse           150     $15
        103     Keyboard        80      $45
        104     Monitor         35      $299
        105     Headphones      60      $79
        """
        
        ax.text(0.5, 0.5, inventory_text, 
                ha='center', va='center', fontsize=11,
                family='monospace', transform=ax.transAxes)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print(f"✅ Created combined test PDF: {pdf_path}")
    return pdf_path


def main():
    """Generate all test PDFs"""
    print("\n" + "="*60)
    print("Generating Test PDFs for DocXtract")
    print("="*60 + "\n")
    
    # Generate test PDFs
    charts_pdf = create_test_pdf_with_charts()
    tables_pdf = create_test_pdf_with_tables()
    combined_pdf = create_combined_test_pdf()
    
    print("\n" + "="*60)
    print("✅ All test PDFs generated successfully!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  1. {charts_pdf} - Bar, line, pie, and scatter charts")
    print(f"  2. {tables_pdf} - Structured and unstructured tables")
    print(f"  3. {combined_pdf} - Mixed tables and charts")
    print(f"\nYou can now test DocXtract with these files:")
    print(f"  python test_data_extraction.py")
    print(f"  python run_dashboard.py")
    print()


if __name__ == "__main__":
    main()
