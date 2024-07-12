"use client";

import { Company, useCompanySymbolQuery } from "@/utils/api";
import { debounce } from "lodash";
import { ChangeEvent, useEffect, useRef, useState } from "react";
import { Input } from "../ui/input";
import { AutoComplete, type Option } from "./Autocomplete";
import { AutoCompleteLazy } from "./AutocompleteLazy";

export const LazyFormInput = ({ className }: { className?: string }) => {
  const [companies, setCompanies] = useState<Company[]>([]);

  //   async function search(searchParams: string) {
  //     // not actually a hook, just an api call
  //     // eslint-disable-next-line react-hooks/rules-of-hooks
  //     const res = await useCompanySymbolQuery({ query: searchParams });
  //     if (!res) return [];
  //     return res;
  //   }

  //   const debouncedSearch = useRef(
  //     debounce(async (criteria) => {
  //       setCompanies(await search(criteria));
  //     }, 300)
  //   ).current;

  //   useEffect(() => {
  //     return () => {
  //       debouncedSearch.cancel();
  //     };
  //   }, [debouncedSearch]);

  //   useEffect(() => {
  //     console.table(companies);
  //   }, [companies]);

  //   async function handleChange(e: ChangeEvent<HTMLInputElement>) {
  //     debouncedSearch(e.target.value);
  //   }

  return (
    <div className={className}>
      <AutoCompleteLazy
        placeholder="Search for a company"
        // onValueChange={setValue}
        // options={companies.map((company) =>
        //   Object({ value: company.symbol, label: company.name })
        // )}
        emptyMessage="No companies found"
        // value={value}
        disabled={false}
      />
    </div>
  );
};
