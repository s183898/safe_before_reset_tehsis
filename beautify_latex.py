



def model_out(path = "model_output.tex"):
    with open(path, 'r') as f:
        lines = f.readlines()
        p_index = 4
        for line in lines:
            line = line.replace(" ", "")
            line = line.replace('\\textbf{C(Method,Treatment(reference="normal"', 'Method: ')
            line = line.replace('\\textbf{C(ethnicity,Treatment(reference="caucasian"', 'Ethnicity: ')
            line = line.replace('\\textbf{C(gender\\_collapsed,Treatment(reference="female"', 'Gender: ')
            line = line.replace('))[T.', '')
            line = line.replace(']}', '')

            values = line.split('&')
            significant = False
            if len(values) > 4:
                if "\\textbf{P$>|$z$|$}" not in values[4]:
                    try:
                        p_value = float(values[p_index])
                        if p_value < 0.05:
                            significant = True
                    except ValueError:
                        pass

            if significant:
                values[p_index] = '\\textbf{' + values[p_index] + '}'
                values[0] = '\\textbf{' + values[p_index] + '}'

            line = '&'.join(values)

            print(line, end='')


import re

def bold_p_values(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    with open(output_file, 'w') as file:
        for line in lines:
            # Find the P-value column (last column in the line)
            parts = line.strip().split('&')
            if len(parts) == 4:
                occupation = parts[0].strip().replace("\\","")
                if "multirow" in occupation:
                    print("Occupation: ", occupation)
                    occupation.replace("{*}{", "\\\\{*}{")

                p_value = parts[-1].strip().replace("\\","")
                if "P-value" in p_value or p_value == '':
                    # Skip the header line
                    file.write(line)
                    continue

                if float(p_value) < 0.05:
                    # Check if it's a numeric value (i.e., it's a valid P-value)
                    parts[-1] = '\\textbf{' + p_value + '} \\\\'
                    line = ' & '.join(parts) + '\n'
            file.write(line)

if __name__ == '__main__':
        
    # Usage example:
    input_file = 'results/chi_squared_results_small.tex'
    output_file = 'results/test_results_paper_collapsed.tex'
    bold_p_values(input_file, output_file)

