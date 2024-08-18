import sys
import json
import logistic

def main():
    if len(sys.argv) <= 1:
        print("Error: Usage: main.py <list-data> <function-value>")
        return
    
    asset = []
    for i in range(0,len(sys.argv)-2):
        asset.append(sys.argv[i+1] * 1)
    function_name = sys.argv[len(sys.argv)-1]

    if function_name == 'logistic':
        result = logistic.classified_image(asset)
        print(json.dumps(result))
    else:
        print(f"Function {function_name} is not recognized")

if __name__ == "__main__":
    main()
