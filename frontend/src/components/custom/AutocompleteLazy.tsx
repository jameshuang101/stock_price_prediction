import {
  CommandGroup,
  CommandItem,
  CommandList,
  CommandInput,
} from "../ui/command";
import { Command as CommandPrimitive } from "cmdk";
import {
  useState,
  useRef,
  useCallback,
  type KeyboardEvent,
  useEffect,
  ChangeEvent,
} from "react";
import { Skeleton } from "../ui/skeleton";
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { Company, useCompanySymbolQuery } from "@/utils/api";
import { useDebounce } from "react-use";

export type Option = Record<"value" | "label", string> & Record<string, string>;

type AutoCompleteLazyProps = {
  emptyMessage: string;
  value?: Option;
  onValueChange?: (value: Option) => void;
  isLoading?: boolean;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
};

export const AutoCompleteLazy = ({
  placeholder,
  emptyMessage,
  value,
  onValueChange,
  disabled,
  isLoading = false,
  className,
}: AutoCompleteLazyProps) => {
  const inputRef = useRef<HTMLInputElement>(null);

  const [isOpen, setOpen] = useState(false);
  const [selected, setSelected] = useState<Option>(value as Option);
  const [state, setState] = useState("typing stopped");
  const [inputValue, setInputValue] = useState<string>(value?.label || "");
  const [options, setOptions] = useState<Option[]>([]);
  const [debouncedValue, setDebouncedValue] = useState("");
  const [_, cancel] = useDebounce(
    async () => {
      setState("typing stopped");
      setDebouncedValue(inputValue);
      const res = await search(inputValue);
      if (!res) {
        setOptions([]);
      } else {
        setOptions(
          res.map((company) =>
            Object({ value: company.symbol, label: company.name })
          )
        );
      }
    },
    500,
    [inputValue]
  );

  async function search(searchParams: string) {
    // not actually a hook, just an api call
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const res = await useCompanySymbolQuery({ query: searchParams });
    if (!res) return [];
    return res;
  }

  // const debouncedSearch = useRef(
  //   debounce(async (criteria) => {
  //     setOptions(
  //       (await search(criteria)).map((company) =>
  //         Object({ value: company.symbol, label: company.name })
  //       )
  //     );
  //   }, 300)
  // ).current;

  // useEffect(() => {
  //   return () => {
  //     debouncedSearch.cancel();
  //   };
  // }, [debouncedSearch]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const input = inputRef.current;
      if (!input) {
        return;
      }

      // debouncedSearch(input.value);
      setState("typing...");
      setInputValue(input.value);

      // Keep the options displayed when the user is typing
      if (!isOpen) {
        setOpen(true);
      }

      // This is not a default behaviour of the <input /> field
      if (event.key === "Enter" && input.value !== "") {
        const optionToSelect = options.find(
          (option) => option.label === input.value
        );
        if (optionToSelect) {
          setSelected(optionToSelect);
          onValueChange?.(optionToSelect);
        }
      }

      if (event.key === "Escape") {
        input.blur();
      }
    },
    [isOpen, options, onValueChange]
  );

  const handleBlur = useCallback(() => {
    setOpen(false);
    setInputValue(selected?.label);
  }, [selected]);

  const handleSelectOption = useCallback(
    (selectedOption: Option) => {
      setInputValue(selectedOption.label);

      setSelected(selectedOption);
      onValueChange?.(selectedOption);

      // This is a hack to prevent the input from being focused after the user selects an option
      // We can call this hack: "The next tick"
      setTimeout(() => {
        inputRef?.current?.blur();
      }, 0);
    },
    [onValueChange]
  );

  return (
    <CommandPrimitive onKeyDown={handleKeyDown}>
      <div>
        <CommandInput
          ref={inputRef}
          value={inputValue}
          onValueChange={isLoading ? undefined : setInputValue}
          onBlur={handleBlur}
          onFocus={() => setOpen(true)}
          placeholder={placeholder}
          disabled={disabled}
          className="text-base"
        />
      </div>
      <div className="relative mt-1">
        <div
          className={cn(
            "animate-in fade-in-0 zoom-in-95 absolute top-0 z-10 w-full rounded-xl bg-white outline-none",
            isOpen ? "block" : "hidden"
          )}
        >
          <CommandList className="rounded-lg ring-1 ring-slate-200">
            {isLoading ? (
              <CommandPrimitive.Loading>
                <div className="p-1">
                  <Skeleton className="w-full h-8" />
                </div>
              </CommandPrimitive.Loading>
            ) : null}
            {options.length > 0 && !isLoading ? (
              <CommandGroup>
                {options.map((option) => {
                  const isSelected = selected?.value === option.value;
                  return (
                    <CommandItem
                      key={option.value}
                      value={option.label}
                      onMouseDown={(event) => {
                        setSelected(option);
                        setInputValue(option.value);
                        event.preventDefault();
                        event.stopPropagation();
                      }}
                      onSelect={() => handleSelectOption(option)}
                      className={cn("flex w-full items-center gap-2")}
                    >
                      <div className="flex flex-col">
                        <p className="text-lg font-medium">{option.value}</p>
                        {option.label}
                      </div>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            ) : null}
            {!isLoading ? (
              <CommandPrimitive.Empty className="px-2 py-3 text-sm text-center rounded-sm select-none">
                {emptyMessage}
              </CommandPrimitive.Empty>
            ) : null}
          </CommandList>
        </div>
      </div>
    </CommandPrimitive>
  );
};
