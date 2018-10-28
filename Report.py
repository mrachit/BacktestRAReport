import pandas as pd
from ras.core.reporting.ReportParameters import ReportParameters
from ras.core.reporting.static_file_parameters import StaticFileParameters
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle, \
    PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle as PS
from PyPDF2 import PdfFileMerger, PdfFileReader
from reportlab.pdfgen.canvas import Canvas
from pdfrw import PdfReader
from pdfrw.toreportlab import makerl
from pdfrw.buildxobj import pagexobj
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_RIGHT, TA_CENTER


class Report:

    @staticmethod
    def table_to_pdf(
            data_2d_list,
            output_location_filename='',
            heading_text='',
            heading_font_size=14,
            heading_leading_spaces=16,
            heading_space_after=20,
            number_of_rows_to_repeat_next_page=1,
            page_size=letter,
            footer_font_size=8,
            footer_space_before=0,
            footer_space_after=5,
            footer_text=None,
            col_widths: list = None,
            add_tbl_styles: list = None,
            alternate_background_rows=1,
            build_flag=True,
            data_align=TA_RIGHT,
            column_align=TA_CENTER

    ):
        """
        Converts a 2d lIst into a Tabular PDF
        Args:
            data_2d_list:
            output_location_filename:
            heading_text:
            heading_font_size:
            heading_leading_spaces:
            heading_space_after:
            number_of_rows_to_repeat_next_page:
            page_size:
            footer_font_size:
            footer_space_before:
            footer_space_after:
            footer_text:
            col_widths:
            add_tbl_styles:
            build_flag:

        Returns:

        """

        temp_list = [

            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            # ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('BOX', (0, 1), (-1, -1), 0.75, colors.black),
            ('BOX', (0, 0), (-1, 0), 0.75, colors.black)
        ]

        styles = getSampleStyleSheet()

        style_data = PS(name='right', parent=styles[
            'Normal'], alignment=data_align)

        style_column = PS(name='right', parent=styles[
            'Normal'], alignment=column_align)

        normal_style = styles['Normal']

        # container for the 'Flowable' objects
        elements = []
        h1 = PS(
            name='Heading1',
            fontSize=heading_font_size,
            leading=heading_leading_spaces,
            spaceAfter=heading_space_after
        )

        f1 = PS(
            name='Footer 1',
            fontSize=footer_font_size,
            spaceBefore=footer_space_before,
            spaceAfter=footer_space_after
        )

        # Use Paragraph flowable container for each element to word wrap it in the Table
        data = []
        # convert to paragraph (center aligned header and first column
        for i in range(number_of_rows_to_repeat_next_page):
            data.append([Paragraph(x, style_column) for x in data_2d_list[i]])

        # convert to paragraph ( Right aligned the data)
        for i in range(number_of_rows_to_repeat_next_page, len(data_2d_list)):
            data.append([Paragraph(x, style_data) for x in data_2d_list[i]])

        # alternate colored rows
        data_len = len(data)

        for each in range(data_len - number_of_rows_to_repeat_next_page):
            if int(each / alternate_background_rows) % 2 == 0:
                bg_color = colors.whitesmoke
            else:
                bg_color = colors.lightgrey

            temp_list.append(('BACKGROUND',
                              (0, each + number_of_rows_to_repeat_next_page),
                              (-1, each + number_of_rows_to_repeat_next_page),
                              bg_color))

        t = Table(data=data,
                  repeatRows=number_of_rows_to_repeat_next_page,
                  colWidths=col_widths)

        if add_tbl_styles is not None:
            temp_list = temp_list + add_tbl_styles

        # border
        table_border_style = TableStyle(temp_list)
        t.setStyle(tblstyle=table_border_style)

        elements.append(Paragraph(heading_text.replace('\n', '<br />\n'), h1))
        elements.append(t)
        if footer_text is not None:
            elements.append(
                Paragraph(footer_text.replace('\n', '<br />\n'), f1))
        # write the document to disk

        if build_flag:
            doc = SimpleDocTemplate(output_location_filename,
                                    pagesize=landscape(page_size),
                                    rightMargin=inch,
                                    leftMargin=inch,
                                    topMargin=inch,
                                    bottomMargin=inch)
            doc.build(elements)
            return None
        else:
            elements.append(PageBreak())
            return elements

    @staticmethod
    def merge_pdf_files(
            filename_list: list,
            output_location_filename: str,
            logo_flag: bool = True,
            page_number_flag: bool = True,
            source_data_footext=ReportParameters.REFERENCE_DATA_TEXT,
            reference_footer_text=ReportParameters.REFERENCE_USE_FOOTER_TEXT
    ):
        """
        Given a list of pdf file names create a merged pdf file. Options to
        add logo and pagenumber

        Args:
            filename_list: List of PDF file names
            output_location_filename: PDF output filename

        Returns:
            None

        """
        merger = PdfFileMerger()
        for filename in filename_list:
            f = open(filename, 'rb')
            merger.append(PdfFileReader(f))
            f.close()
        merger.write(output_location_filename)
        if logo_flag:
            Report.add_logo(output_location_filename)
        if page_number_flag:
            Report.add_page_number(
                pdf_name=output_location_filename,
                source_data_footext=source_data_footext,
                reference_footer_text=reference_footer_text
            )

    @staticmethod
    def get_number_of_pages_pdf(filename: str) -> int:
        """
        Gets the number of pages in a pdf file
        Args:
            filename: pdf filename

        Returns:
            Number of pages in a pdf file

        """
        f = open(filename, 'rb')
        return PdfFileReader(f).getNumPages()


    @staticmethod
    def create_toc_pdf(
            output_file_name: str,
            title_text: str,
            content_name_dict,
            page_number_list: list,
            page_layout=landscape(pagesize=letter)):
        """

        Args:
            output_file_name:
            title_text:
            content_name_dict:

        Returns:

        """
        title = PS(
            name='Title 1',
            fontSize=14,
            leading=16
        )

        h1 = PS(
            name='Header 1',
            fontSize=12
        )
        h1_pagenumber = PS(
            name='Header 1',
            fontSize=12,
            alignment=TA_RIGHT
        )

        h2_indented = PS(
            name='Header 2',
            fontSize=10,
            spaceBefore=4,
            leftIndent=20
        )
        h2 = PS(
            name='Header 2',
            fontSize=10,
            spaceBefore=4,
            alignment=TA_RIGHT
        )

        elements = []

        elements.append(
            Paragraph('<b>' + title_text + '</b> <br /><br />', style=title))

        data = []
        i = 0

        for key in content_name_dict:
            data.append(
                [Paragraph('<i><b>' + str(key) + '</b></i>', style=h1),
                 Paragraph('<b>' + str(page_number_list[i]) + '</b>',
                           style=h1_pagenumber)]
            )
            for value in content_name_dict[key]:
                data.append(
                    [
                        Paragraph(value + '<br />', style=h2_indented),
                        Paragraph(str(page_number_list[i]), style=h2)
                    ])
                i = i + 1

        t = Table(data=data,
                  colWidths=[8 * inch] + [1 * inch])

        elements.append(t)

        doc = SimpleDocTemplate(output_file_name,
                                pagesize=page_layout,
                                rightMargin=inch,
                                leftMargin=inch,
                                topMargin=inch,
                                bottomMargin=inch)

        doc.build(elements)

    @staticmethod
    def create_disclosures_pdf(
            output_filename,
            excel_filename=StaticFileParameters.DISCLOSURES_FILENAME,
            excel_tab_name="Disclosure",
            page_layout=landscape(letter)
    ):
        """

        Args:
            output_filename:
            page_size:

        Returns:

        """
        a = pd.read_excel(
            io=excel_filename,
            sheet_name=excel_tab_name)

        disclosures_txt = 'Disclosures: \n'
        for i in range(len(a)):
            disclosures_txt = disclosures_txt + a.loc[i][0] + '\n'

        f1 = PS(
            name='Footer 1',
            fontSize=7
        )
        doc = SimpleDocTemplate(output_filename,
                                pagesize=page_layout,
                                rightMargin=inch,
                                leftMargin=inch,
                                topMargin=inch,
                                bottomMargin=inch
                                )

        doc.build([Paragraph(text=disclosures_txt.replace('\n', '<br />\n'),
                             style=f1)])



    @staticmethod
    def add_page_number(
            pdf_name,
            source_data_footext=ReportParameters.REFERENCE_DATA_TEXT,
            reference_footer_text=ReportParameters.REFERENCE_USE_FOOTER_TEXT
    ):
        """

        Args:
            pdf_name:
            source_data_footext:
            reference_footer_text:

        Returns:

        """

        input_file = pdf_name
        output_file = pdf_name

        # Get pages
        reader = PdfReader(input_file)
        pages = [pagexobj(p) for p in reader.pages]

        # Compose new pdf
        canvas = Canvas(output_file)

        for page_num, page in enumerate(pages, start=1):
            # Add page
            canvas.setPageSize((page.BBox[2], page.BBox[3]))
            canvas.doForm(makerl(canvas, page))

            # Draw footer
            page_number_footer_text = "Page %s of %s" % (page_num, len(pages))

            x = 128
            canvas.saveState()
            # canvas.setStrokeColorRGB(0, 0, 0)
            canvas.setLineWidth(0.5)
            canvas.line(66, 72, page.BBox[2] - 66, 72)
            canvas.setFont('Times-Roman', 10)
            canvas.drawString(x=(page.BBox[2] - x),
                              y=60, text=page_number_footer_text)

            canvas.drawString(x=66,
                              y=60, text=source_data_footext)

            canvas.drawString(x=66, y=45,
                              text=reference_footer_text)

            canvas.restoreState()
            canvas.showPage()

        # Save pdf
        canvas.save()

    @staticmethod
    def get_final_pdf(
            elements: list,
            output_location_filename: str,
            page_layout=landscape(letter)
    ):
        """
        Given a List of Reportlab elemnts build a PDF
        Args:
            elements:
            output_location_filename:
            page_layout:

        Returns:

        """
        doc = SimpleDocTemplate(
            output_location_filename,
            pagesize=page_layout,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch, bottomMargin=inch)

        doc.build(elements)

    @staticmethod
    def get_footer_text_formatted(
            footer_text,
            footer_font_size=6,
            footer_space_before=10,
            footer_space_after=5
                                  ):
        """
        Helper method to get the Reportlab Paragraph formatting on the footer
        text

        :param footer_text:
        :param footer_font_size:
        :param footer_space_before:
        :param footer_space_after:
        :return:
        """
        f1 = PS(
            name='Footer 1',
            fontSize=footer_font_size,
            spaceBefore=footer_space_before,
            spaceAfter=footer_space_after
        )
        return Paragraph(footer_text.replace('\n', '<br />\n'), f1)


    @staticmethod
    def add_logo(pdf_name: str,):
        """
        Adds logo at the top for all pages in the pdf name

        Args:
            pdf_name:

        Returns:
            None. Updates the input pdf
        """

        input_file = pdf_name
        output_file = pdf_name

        # Get pages
        reader = PdfReader(input_file)
        pages = [pagexobj(p) for p in reader.pages]

        # Compose new pdf
        canvas = Canvas(output_file)

        for page_num, page in enumerate(pages, start=1):
            # Add page
            canvas.setPageSize((page.BBox[2], page.BBox[3]))
            canvas.doForm(makerl(canvas, page))

            canvas.saveState()
            # canvas.setStrokeColorRGB(0, 0, 0)
            canvas.restoreState()

            # Add Logo
            canvas.drawImage(StaticFileParameters.ra_logo_location, 650, 550,
                             111.750, 39)
            canvas.showPage()

        # Save pdf
        canvas.save()

